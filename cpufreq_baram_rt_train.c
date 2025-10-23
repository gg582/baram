/*
 * drivers/cpufreq/cpufreq_baram.c
 * baram: laptop-oriented conservative governor
 *
 * Conservative-style governor backed by a lightweight 1D CNN inference engine
 * that can also adapt its parameters online inside the driver.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kernel_stat.h>
#include <linux/power_supply.h>
#include <linux/slab.h>
#include <linux/cpufreq.h>
#include <linux/workqueue.h>
#include <linux/jiffies.h>
#include <linux/sysfs.h>
#include <linux/kobject.h>
#include <linux/cpumask.h>
#include <linux/cpu.h>
#include <linux/tick.h>
#include <linux/mutex.h>
#include <linux/sched/cpufreq.h>
#include <linux/errno.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/math64.h>
#include "include/ac_names_gen.h"
#include "include/battery.h"

#define FP_SHIFT 16
#define FP_SCALE (1 << FP_SHIFT)

#define CNN_Q 10
#define CNN_ONE (1 << CNN_Q)
#define CNN_WINDOW 16
#define CNN_KERNEL_SIZE 3
#define CNN_CONV1_OUT 2
#define CNN_CONV2_OUT 1

#define CNN_MAX_VALUE  32767
#define CNN_MIN_VALUE -32768

#define LAP_DEF_LEARNING_RATE_FP (FP_SCALE / 5)
#define LAP_MAX_LEARNING_RATE_FP (FP_SCALE)

#define LAP_DEF_TARGET_LOAD 50

// Dynamic tuning values for AC vs. Battery
#define LAP_AC_LEARNING_RATE_FP (FP_SCALE / 4)  // 0.25 gain on AC
#define LAP_BATTERY_LEARNING_RATE_FP (FP_SCALE / 8) // 0.125 gain on battery
#define LAP_AC_FREQ_STEP 10
#define LAP_BATTERY_FREQ_STEP 3
#define LAP_AC_TARGET_LOAD 70
#define LAP_BATTERY_TARGET_LOAD 30

#define LAP_HIGH_LOAD_BYPASS 95
#define LAP_LOW_LOAD_BYPASS   5

#define LAP_TRAIN_RATE_SHIFT 12
#define LAP_TRAIN_BIAS_SHIFT 8

struct lap_cpu_dbs {
    u64 prev_cpu_idle;
    u64 prev_cpu_nice;
    u64 prev_update_time;
};

DEFINE_PER_CPU(struct lap_cpu_dbs, lap_cpu_dbs);

// Global data for system-wide load calculation
static DEFINE_MUTEX(lap_global_lock);
static unsigned int lap_global_load;
static bool lap_on_ac_power;

static DEFINE_MUTEX(lap_cnn_weight_lock);

struct lap_tuners {
    unsigned int freq_step;
    unsigned int sampling_down_factor;
    unsigned int ignore_nice_load;
    unsigned int sampling_rate;
    s64 learning_rate_fp;
    unsigned int target_load;
};

struct lap_cnn_state {
    s16 history[CNN_WINDOW];
};

struct lap_policy_info {
    struct cpufreq_policy *policy;
    unsigned int requested_freq;
    struct lap_tuners tuners;
    struct lap_cnn_state cnn;
    unsigned int last_target_load;
    struct delayed_work work;
    struct mutex lock;
    bool cnn_has_prediction;
    s16 cnn_last_prediction;
    s32 cnn_last_avg;
    bool cnn_has_history;
    s16 cnn_smoothed_output;
};

#if CNN_CONV2_OUT != 1
#error "Current implementation expects a single output channel"
#endif

static s16 lap_cnn_conv1_weights[CNN_CONV1_OUT][CNN_KERNEL_SIZE] = {
    { -256, 0, 256 },
    { 128, 256, 128 },
};

static s16 lap_cnn_conv1_bias[CNN_CONV1_OUT] = { 0, 0 };

static s16 lap_cnn_conv2_weights[CNN_CONV2_OUT][CNN_CONV1_OUT][CNN_KERNEL_SIZE] = {
    {
        { -192, 0, 192 },
        { 64, 128, 64 },
    },
};

static s16 lap_cnn_conv2_bias[CNN_CONV2_OUT] = { 0 };

static s16 lap_cnn_fc_weight = 768; /* ~0.75 in Q10 */
static s16 lap_cnn_fc_bias = 0;

#define LAP_DEF_FREQ_STEP          5
#define LAP_MAX_FREQ_STEP_PERCENT  25
#define LAP_MIN_FREQ_STEP_PERCENT  5
#define LAP_DEF_SAMPLING_DOWN_FAC  2
#define LAP_MAX_SAMPLING_DOWN_FAC  5
#define LAP_DEF_SAMPLING_RATE      1

/* Function Prototypes */
static inline unsigned int lap_get_freq_step_khz(struct lap_tuners *tuners, struct cpufreq_policy *policy);
static unsigned int lap_dbs_get_load(bool ignore_nice);
static s16 lap_cnn_scale_load(unsigned int load, unsigned int target);
static void lap_cnn_init(struct lap_cnn_state *state, s16 initial_sample);
static void lap_cnn_reset_state(struct lap_policy_info *lp, s16 initial_sample);
static void lap_cnn_push(struct lap_cnn_state *state, s16 sample);
static s16 lap_cnn_predict(const struct lap_cnn_state *state, s32 *avg_out);
static void lap_cnn_train(struct lap_policy_info *lp, s16 actual_sample);
static void lap_apply_cnn_policy(struct cpufreq_policy *policy, struct lap_policy_info *lp, unsigned int load, s16 cnn_sample);
static unsigned long cs_dbs_update(struct cpufreq_policy *policy);
static void lap_work_handler(struct work_struct *work);

// Sysfs functions
static ssize_t show_sampling_rate(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
static ssize_t store_sampling_rate(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
static ssize_t show_sampling_down_factor(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
static ssize_t store_sampling_down_factor(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
static ssize_t show_ignore_nice_load(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
static ssize_t store_ignore_nice_load(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
static ssize_t show_freq_step(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
static ssize_t store_freq_step(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
static ssize_t show_learning_rate(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
static ssize_t store_learning_rate(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
static ssize_t show_target_load(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
static ssize_t store_target_load(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count);
static int lap_start(struct cpufreq_policy *policy);
static void lap_stop(struct cpufreq_policy *policy);
static int lap_init(struct cpufreq_policy *policy);
static void lap_exit(struct cpufreq_policy *policy);
static int __init baram_module_init(void);
static void __exit baram_module_exit(void);

static inline s16 lap_cnn_clamp(s32 value)
{
    if (value > CNN_MAX_VALUE)
        value = CNN_MAX_VALUE;
    else if (value < CNN_MIN_VALUE)
        value = CNN_MIN_VALUE;
    return (s16)value;
}

static inline s16 lap_cnn_activate(s32 value)
{
    if (value >= 0) {
        if (value > CNN_MAX_VALUE)
            value = CNN_MAX_VALUE;
        return (s16)value;
    }

    value >>= 2; /* Leaky behaviour for negative inputs */
    if (value < CNN_MIN_VALUE)
        value = CNN_MIN_VALUE;
    if (value > CNN_MAX_VALUE)
        value = CNN_MAX_VALUE;
    return (s16)value;
}

/* lap_get_freq_step_khz - Compute freq step in kHz from percent */
static inline unsigned int lap_get_freq_step_khz(struct lap_tuners *tuners, struct cpufreq_policy *policy)
{
    unsigned int step_khz;
    step_khz = (tuners->freq_step * policy->max) / 100;
    if (unlikely(step_khz == 0)) {
        step_khz = (LAP_MIN_FREQ_STEP_PERCENT * policy->max) / 100;
    }
    if (unlikely(step_khz == 0))
        step_khz = 1;
    return step_khz;
}

/* lap_dbs_get_load - compute average load (0..100) across all online CPUs */
static unsigned int lap_dbs_get_load(bool ignore_nice)
{
    unsigned int load_sum = 0;
    unsigned int cpu;
    u64 cur_time;
    unsigned int time_elapsed;
    unsigned int cur_load;
    u64 cur_idle, cur_nice;
    u64 idle_delta, nice_delta;

    for_each_online_cpu(cpu) {
        struct lap_cpu_dbs *cdbs = per_cpu_ptr(&lap_cpu_dbs, cpu);
        cur_idle = get_cpu_idle_time_us(cpu, &cur_time);
        cur_nice = jiffies_to_usecs(kcpustat_cpu(cpu).cpustat[CPUTIME_NICE]);
        time_elapsed = (unsigned int)(cur_time - cdbs->prev_update_time);
        idle_delta = (unsigned int)(cur_idle - cdbs->prev_cpu_idle);
        nice_delta = (unsigned int)(cur_nice - cdbs->prev_cpu_nice);

        if (unlikely(time_elapsed == 0)) {
            cur_load = 100;
        } else {
            unsigned int busy_time = time_elapsed - idle_delta;
            if (ignore_nice)
                busy_time -= nice_delta;
            cur_load = 100 * busy_time / time_elapsed;
        }

        cdbs->prev_cpu_idle = cur_idle;
        cdbs->prev_cpu_nice = cur_nice;
        cdbs->prev_update_time = cur_time;

        load_sum += cur_load;
    }

    if (unlikely(num_online_cpus() == 0))
        return 0;

    return load_sum / num_online_cpus();
}

/* lap_is_on_ac - Retrieves AC status and updates governor state */
static void lap_is_on_ac(void)
{
    struct power_supply *psy;
    union power_supply_propval val;
    int i;

    lap_on_ac_power = false;

    for (i = 0; i < ARRAY_SIZE(ac_names) && ac_names[i] != NULL; i++) {
        psy = power_supply_get_by_name(ac_names[i]);
        if (!psy)
            continue;

        if (power_supply_get_property(psy, POWER_SUPPLY_PROP_ONLINE, &val) == 0 && val.intval) {
            lap_on_ac_power = true;
        }
        power_supply_put(psy);

        if (lap_on_ac_power) {
            return;
        }
    }
}

static s16 lap_cnn_scale_load(unsigned int load, unsigned int target)
{
    s64 diff = (s64)load - target;
    s64 scaled = diff * CNN_ONE;

    scaled = div_s64(scaled, 100);
    scaled = clamp_t(s64, scaled, CNN_MIN_VALUE, CNN_MAX_VALUE);

    return (s16)scaled;
}

static void lap_cnn_init(struct lap_cnn_state *state, s16 initial_sample)
{
    int i;

    for (i = 0; i < CNN_WINDOW; i++)
        state->history[i] = initial_sample;
}

static void lap_cnn_reset_state(struct lap_policy_info *lp, s16 initial_sample)
{
    lap_cnn_init(&lp->cnn, initial_sample);
    lp->cnn_has_prediction = false;
    lp->cnn_last_prediction = initial_sample;
    lp->cnn_last_avg = 0;
    lp->cnn_has_history = false;
    lp->cnn_smoothed_output = initial_sample;
}

static void lap_cnn_push(struct lap_cnn_state *state, s16 sample)
{
    memmove(&state->history[0], &state->history[1],
        (CNN_WINDOW - 1) * sizeof(state->history[0]));
    state->history[CNN_WINDOW - 1] = sample;
}

static s16 lap_cnn_predict(const struct lap_cnn_state *state, s32 *avg_out)
{
    s16 conv1_out[CNN_CONV1_OUT][CNN_WINDOW];
    s16 conv2_out[CNN_WINDOW];
    s32 acc;
    int pos, k, oc, ic;
    int pad = CNN_KERNEL_SIZE / 2;
    s16 sample;
    s64 sum = 0;
    s64 avg64;
    s32 avg32;
    s32 fc_acc;

    for (oc = 0; oc < CNN_CONV1_OUT; oc++) {
        for (pos = 0; pos < CNN_WINDOW; pos++) {
            acc = (s32)lap_cnn_conv1_bias[oc] << CNN_Q;
            for (k = 0; k < CNN_KERNEL_SIZE; k++) {
                int idx = pos + k - pad;

                if (idx < 0)
                    sample = state->history[0];
                else if (idx >= CNN_WINDOW)
                    sample = state->history[CNN_WINDOW - 1];
                else
                    sample = state->history[idx];

                acc += (s32)lap_cnn_conv1_weights[oc][k] * sample;
            }

            conv1_out[oc][pos] = lap_cnn_activate(acc >> CNN_Q);
        }
    }

    for (pos = 0; pos < CNN_WINDOW; pos++) {
        acc = (s32)lap_cnn_conv2_bias[0] << CNN_Q;
        for (ic = 0; ic < CNN_CONV1_OUT; ic++) {
            for (k = 0; k < CNN_KERNEL_SIZE; k++) {
                int idx = pos + k - pad;

                if (idx < 0)
                    sample = conv1_out[ic][0];
                else if (idx >= CNN_WINDOW)
                    sample = conv1_out[ic][CNN_WINDOW - 1];
                else
                    sample = conv1_out[ic][idx];

                acc += (s32)lap_cnn_conv2_weights[0][ic][k] * sample;
            }
        }

        conv2_out[pos] = lap_cnn_clamp(acc >> CNN_Q);
        sum += conv2_out[pos];
    }

    avg64 = div_s64(sum, CNN_WINDOW);
    avg64 = clamp_t(s64, avg64, CNN_MIN_VALUE, CNN_MAX_VALUE);
    avg32 = (s32)avg64;

    if (avg_out)
        *avg_out = avg32;

    fc_acc = ((s32)lap_cnn_fc_bias << CNN_Q) + (s32)lap_cnn_fc_weight * avg32;
    return lap_cnn_clamp(fc_acc >> CNN_Q);
}

static void lap_cnn_train(struct lap_policy_info *lp, s16 actual_sample)
{
    s32 error;
    s32 delta_w;
    s32 delta_b;
    s32 new_weight;
    s32 new_bias;

    if (!lp->cnn_has_prediction)
        return;

    error = (s32)actual_sample - (s32)lp->cnn_last_prediction;
    if (error == 0)
        goto out_clear;

    mutex_lock(&lap_cnn_weight_lock);

    delta_w = (s32)(((s64)error * lp->cnn_last_avg) >> (CNN_Q + LAP_TRAIN_RATE_SHIFT));
    delta_b = (s32)(((s64)error) >> LAP_TRAIN_BIAS_SHIFT);

    new_weight = (s32)lap_cnn_fc_weight + delta_w;
    new_bias = (s32)lap_cnn_fc_bias + delta_b;

    lap_cnn_fc_weight = (s16)clamp_t(s32, new_weight, CNN_MIN_VALUE, CNN_MAX_VALUE);
    lap_cnn_fc_bias = (s16)clamp_t(s32, new_bias, CNN_MIN_VALUE, CNN_MAX_VALUE);

    mutex_unlock(&lap_cnn_weight_lock);

out_clear:
    lp->cnn_has_prediction = false;
}

static void lap_apply_cnn_policy(struct cpufreq_policy *policy,
                 struct lap_policy_info *lp, unsigned int load, s16 cnn_sample)
{
    unsigned int requested_freq = lp->requested_freq;
    unsigned int step_khz = lap_get_freq_step_khz(&lp->tuners, policy);
    s16 cnn_output = 0;
    s32 avg32 = 0;
    s64 scaled_delta;
    s64 delta_khz;
    s32 applied_output;

    lap_cnn_push(&lp->cnn, cnn_sample);

    if (load >= LAP_HIGH_LOAD_BYPASS) {
        requested_freq = policy->max;
        lp->cnn_has_prediction = false;
        lp->cnn_has_history = false;
    } else if (load <= LAP_LOW_LOAD_BYPASS) {
        requested_freq = policy->min;
        lp->cnn_has_prediction = false;
        lp->cnn_has_history = false;
    } else {
        cnn_output = lap_cnn_predict(&lp->cnn, &avg32);

        applied_output = cnn_output;
        if (lp->cnn_has_history) {
            applied_output = (3 * (s32)lp->cnn_smoothed_output + applied_output) >> 2;
            applied_output = (s32)lap_cnn_clamp(applied_output);
        }

        lp->cnn_smoothed_output = lap_cnn_clamp(applied_output);
        lp->cnn_has_history = true;

        scaled_delta = (s64)lp->cnn_smoothed_output * lp->tuners.learning_rate_fp;
        delta_khz = (scaled_delta * step_khz) >> (CNN_Q + FP_SHIFT);

        requested_freq = clamp_val((s64)requested_freq + delta_khz,
                       policy->min, policy->max);

        lp->cnn_last_prediction = cnn_output;
        lp->cnn_last_avg = avg32;
        lp->cnn_has_prediction = true;
    }

    if (requested_freq != lp->requested_freq) {
        cpufreq_driver_target(policy, requested_freq, CPUFREQ_RELATION_L);
        lp->requested_freq = requested_freq;
    }
}

/* cs_dbs_update - apply Laputil decision logic for one policy */
static unsigned long cs_dbs_update(struct cpufreq_policy *policy)
{
    struct lap_policy_info *lp = policy->governor_data;
    struct lap_tuners *tuners;
    unsigned int load;
    s16 cnn_sample;
    bool target_changed = false;

    if (!lp)
        return HZ;

    tuners = &lp->tuners;
    mutex_lock(&lp->lock);

    if (policy->cpu == 0)
        lap_is_on_ac();

    if (lap_on_ac_power) {
        tuners->learning_rate_fp = LAP_AC_LEARNING_RATE_FP;
        tuners->freq_step = LAP_AC_FREQ_STEP;
        tuners->target_load = LAP_AC_TARGET_LOAD;
    } else {
        tuners->learning_rate_fp = LAP_BATTERY_LEARNING_RATE_FP;
        tuners->freq_step = LAP_BATTERY_FREQ_STEP;
        tuners->target_load = LAP_BATTERY_TARGET_LOAD;
    }

    mutex_lock(&lap_global_lock);
    lap_global_load = lap_dbs_get_load(tuners->ignore_nice_load);
    mutex_unlock(&lap_global_lock);

    load = lap_global_load;
    cnn_sample = lap_cnn_scale_load(load, tuners->target_load);

    if (lp->last_target_load != tuners->target_load) {
        lap_cnn_reset_state(lp, cnn_sample);
        lp->last_target_load = tuners->target_load;
        target_changed = true;
    }

    if (!target_changed)
        lap_cnn_train(lp, cnn_sample);

    lap_apply_cnn_policy(policy, lp, load, cnn_sample);

    mutex_unlock(&lp->lock);
    return (unsigned long)tuners->sampling_rate * HZ;
}

// lap_work_handler - Delayed work handler per policy
static void lap_work_handler(struct work_struct *work)
{
    struct lap_policy_info *lp = container_of(work, struct lap_policy_info, work.work);
    struct cpufreq_policy *policy = lp->policy;
    unsigned long delay_jiffies;
    delay_jiffies = cs_dbs_update(policy);
    schedule_delayed_work_on(policy->cpu, &lp->work, delay_jiffies);
}

// sysfs interface for governor tunables
#define lap_gov_attr(_name) \
static struct kobj_attribute _name##_attr = { \
    .attr = { .name = #_name, .mode = 0644 }, \
    .show = show_##_name, \
    .store = store_##_name, \
}

// Show target load.
static ssize_t show_target_load(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    ssize_t ret;
    if (!lp)
        return snprintf(buf, 2, "0\n");
    mutex_lock(&lp->lock);
    ret = snprintf(buf, 12, "%u\n", lp->tuners.target_load);
    mutex_unlock(&lp->lock);
    return ret;
}

// Store target load.
static ssize_t store_target_load(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
{
    unsigned int val;
    int ret;
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    if (!lp)
        return -EINVAL;
    ret = kstrtouint(buf, 10, &val);
    if (ret || val > 100)
        return -EINVAL;
    mutex_lock(&lp->lock);
    lp->tuners.target_load = val;
    lp->last_target_load = val;
    {
        unsigned int current_load;
        s16 reset_sample;

        mutex_lock(&lap_global_lock);
        current_load = lap_global_load;
        mutex_unlock(&lap_global_lock);

        reset_sample = lap_cnn_scale_load(current_load, val);
        lap_cnn_reset_state(lp, reset_sample);
    }
    mutex_unlock(&lp->lock);
    return count;
}

// Show learning rate.
static ssize_t show_learning_rate(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    ssize_t ret;
    if (!lp)
        return snprintf(buf, 2, "0\n");
    mutex_lock(&lp->lock);
    ret = snprintf(buf, 12, "%llu\n", (unsigned long long)lp->tuners.learning_rate_fp * 1000 / FP_SCALE);
    mutex_unlock(&lp->lock);
    return ret;
}

// Store learning rate.
static ssize_t store_learning_rate(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
{
    unsigned int val;
    int ret;
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    if (!lp)
        return -EINVAL;
    ret = kstrtouint(buf, 10, &val);
    if (ret || val == 0 || val > 1000)
        return -EINVAL;
    mutex_lock(&lp->lock);
    lp->tuners.learning_rate_fp = (s64)val * FP_SCALE / 1000;
    mutex_unlock(&lp->lock);
    return count;
}

// Show sampling rate.
static ssize_t show_sampling_rate(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    ssize_t ret;
    if (!lp)
        return snprintf(buf, 2, "0\n");
    mutex_lock(&lp->lock);
    ret = snprintf(buf, 12, "%u\n", lp->tuners.sampling_rate);
    mutex_unlock(&lp->lock);
    return ret;
}

// Store sampling rate.
static ssize_t store_sampling_rate(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
{
    unsigned int val;
    int ret;
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    if (!lp)
        return -EINVAL;
    ret = kstrtouint(buf, 10, &val);
    if (ret || val == 0)
        return -EINVAL;
    mutex_lock(&lp->lock);
    lp->tuners.sampling_rate = val;
    mutex_unlock(&lp->lock);
    return count;
}

// Show sampling down factor.
static ssize_t show_sampling_down_factor(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    ssize_t ret;
    if (!lp)
        return snprintf(buf, 2, "0\n");
    mutex_lock(&lp->lock);
    ret = snprintf(buf, 12, "%u\n", lp->tuners.sampling_down_factor);
    mutex_unlock(&lp->lock);
    return ret;
}

// Store sampling down factor.
static ssize_t store_sampling_down_factor(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
{
    unsigned int val;
    int ret;
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    if (!lp)
        return -EINVAL;
    ret = kstrtouint(buf, 10, &val);
    if (ret || val < 1 || val > LAP_MAX_SAMPLING_DOWN_FAC)
        return -EINVAL;
    mutex_lock(&lp->lock);
    lp->tuners.sampling_down_factor = val;
    mutex_unlock(&lp->lock);
    return count;
}

// Show ignore nice load.
static ssize_t show_ignore_nice_load(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    ssize_t ret;
    if (!lp)
        return snprintf(buf, 3, "0\n");
    mutex_lock(&lp->lock);
    ret = snprintf(buf, 3, "%u\n", lp->tuners.ignore_nice_load);
    mutex_unlock(&lp->lock);
    return ret;
}

// Store ignore nice load.
static ssize_t store_ignore_nice_load(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
{
    unsigned int val;
    int ret;
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    if (!lp)
        return -EINVAL;
    ret = kstrtouint(buf, 10, &val);
    if (ret || val > 1)
        return -EINVAL;
    mutex_lock(&lp->lock);
    lp->tuners.ignore_nice_load = val;
    mutex_unlock(&lp->lock);
    return count;
}

// Show frequency step.
static ssize_t show_freq_step(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    ssize_t ret;
    if (!lp)
        return snprintf(buf, 3, "0\n");
    mutex_lock(&lp->lock);
    ret = snprintf(buf, 12, "%u\n", lp->tuners.freq_step);
    mutex_unlock(&lp->lock);
    return ret;
}

// Store frequency step.
static ssize_t store_freq_step(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
{
    unsigned int val;
    int ret;
    struct cpufreq_policy *policy = container_of(kobj, struct cpufreq_policy, kobj);
    struct lap_policy_info *lp = policy->governor_data;
    if (!lp)
        return -EINVAL;
    ret = kstrtouint(buf, 10, &val);
    if (ret || val < LAP_MIN_FREQ_STEP_PERCENT || val > LAP_MAX_FREQ_STEP_PERCENT)
        return -EINVAL;
    mutex_lock(&lp->lock);
    lp->tuners.freq_step = val;
    mutex_unlock(&lp->lock);
    return count;
}

static struct kobj_attribute learning_rate_attr = {
    .attr = {.name = "learning_rate", .mode = 0644},
    .show = show_learning_rate,
    .store = store_learning_rate,
};

static struct kobj_attribute sampling_rate_attr = {
    .attr = {.name = "sampling_rate", .mode = 0644},
    .show = show_sampling_rate,
    .store = store_sampling_rate,
};

static struct kobj_attribute sampling_down_factor_attr = {
    .attr = {.name = "sampling_down_factor", .mode = 0644},
    .show = show_sampling_down_factor,
    .store = store_sampling_down_factor,
};

static struct kobj_attribute target_load_attr = {
    .attr = {.name = "target_load", .mode = 0644},
    .show = show_target_load,
    .store = store_target_load,
};

static struct kobj_attribute ignore_nice_load_attr = {
    .attr = {.name = "ignore_nice_load", .mode = 0644},
    .show = show_ignore_nice_load,
    .store = store_ignore_nice_load,
};

static struct kobj_attribute freq_step_attr = {
    .attr = {.name = "freq_step", .mode = 0644},
    .show = show_freq_step,
    .store = store_freq_step,
};

static struct attribute *baram_attrs[] = {
    &freq_step_attr.attr,
    &ignore_nice_load_attr.attr,
    &sampling_down_factor_attr.attr,
    &sampling_rate_attr.attr,
    &target_load_attr.attr,
    &learning_rate_attr.attr,
    NULL
};

static struct attribute_group baram_attr_group = {
    .attrs = baram_attrs,
    .name = "baram-rt-train"
};

/**
 * @brief Starts the governor for a policy.
 * @param policy CPU frequency policy.
 * @return 0 on success, negative errno on failure.
 */
static int lap_start(struct cpufreq_policy *policy)
{
    struct lap_policy_info *lp;
    lp = kzalloc(sizeof(*lp), GFP_KERNEL);
    if (!lp)
        return -ENOMEM;
    INIT_DELAYED_WORK(&lp->work, lap_work_handler);
    lp->policy = policy;
    mutex_init(&lp->lock);
    lp->tuners.freq_step = LAP_DEF_FREQ_STEP;
    lp->tuners.sampling_down_factor = LAP_DEF_SAMPLING_DOWN_FAC;
    lp->tuners.ignore_nice_load = 1;
    lp->tuners.sampling_rate = LAP_DEF_SAMPLING_RATE;
    lp->tuners.learning_rate_fp = LAP_DEF_LEARNING_RATE_FP;
    lp->tuners.target_load = LAP_DEF_TARGET_LOAD;
    lp->last_target_load = lp->tuners.target_load;
    lap_cnn_reset_state(lp, 0);
    if (policy->cur)
        lp->requested_freq = policy->cur;
    else
        lp->requested_freq = policy->min;
    policy->governor_data = lp;
    if (sysfs_create_group(&policy->kobj, &baram_attr_group)) {
        kfree(lp);
        policy->governor_data = NULL;
        return -EINVAL;
    }
    schedule_delayed_work_on(policy->cpu, &lp->work, 0);
    return 0;
}

/**
 * @brief Stops the governor for a policy.
 * @param policy CPU frequency policy.
 */
static void lap_stop(struct cpufreq_policy *policy)
{
    struct lap_policy_info *lp = policy->governor_data;
    if (lp) {
        cancel_delayed_work_sync(&lp->work);
        sysfs_remove_group(&policy->kobj, &baram_attr_group);
        kfree(lp);
        policy->governor_data = NULL;
    }
}

/**
 * @brief Initializes the governor for a policy.
 * @param policy CPU frequency policy.
 * @return 0 on success, negative errno on failure.
 */
static int lap_init(struct cpufreq_policy *policy)
{
    return 0;
}

/**
 * @brief Exits the governor for a policy.
 * @param policy CPU frequency policy.
 */
static void lap_exit(struct cpufreq_policy *policy)
{
    return;
}

static struct cpufreq_governor baram_governor = {
    .name = "baram-rt-train",
    .flags = 0,
    .init = lap_init,
    .exit = lap_exit,
    .start = lap_start,
    .stop = lap_stop,
};

static int __init baram_module_init(void)
{
    return cpufreq_register_governor(&baram_governor);
}

static void __exit baram_module_exit(void)
{
    cpufreq_unregister_governor(&baram_governor);
}

MODULE_AUTHOR("Lee Yunjin <gzblues61@daum.net>");
MODULE_DESCRIPTION("'cpufreq_baram(Real-Time Train Edition)' - Conservative-style governor with adaptive 1D CNN");
MODULE_LICENSE("GPL");

module_init(baram_module_init);
module_exit(baram_module_exit);
