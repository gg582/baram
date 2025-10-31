/*
 * drivers/cpufreq/cpufreq_baram.c
 * baram: laptop-oriented conservative governor
 *
 * Copyright (C) 2025 Lee Yunjin <gzblues61@daum.net>
 *
 * Conservative-style governor backed by a lightweight LSTM inference engine.
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

#define LSTM_Q 10
#define LSTM_ONE (1 << LSTM_Q)
#define LSTM_WINDOW 16
#define LSTM_HIDDEN_SIZE 4

#define LSTM_MAX_VALUE  32767
#define LSTM_MIN_VALUE -32768

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

struct lap_tuners {
    unsigned int freq_step;
    unsigned int sampling_down_factor;
    unsigned int ignore_nice_load;
    unsigned int sampling_rate;
    s64 learning_rate_fp;
    unsigned int target_load;
};

struct lap_lstm_state {
    s16 history[LSTM_WINDOW];
    s16 hidden[LSTM_HIDDEN_SIZE];
    s16 cell[LSTM_HIDDEN_SIZE];
};

struct lap_policy_info {
    struct cpufreq_policy *policy;
    unsigned int requested_freq;
    struct lap_tuners tuners;
    struct lap_lstm_state lstm;
    unsigned int last_target_load;
    struct delayed_work work;
    struct mutex lock;
    struct cpumask eff_mask;  /* Efficiency cores mask */
    struct cpumask perf_mask; /* Performance cores mask */
    struct cpumask lp_eff_mask;
};

static const s16 lap_lstm_wg_i[LSTM_HIDDEN_SIZE] = { 128, 128, 128, 128 };
static const s16 lap_lstm_wg_f[LSTM_HIDDEN_SIZE] = { 128, 128, 128, 128 };
static const s16 lap_lstm_wg_c[LSTM_HIDDEN_SIZE] = { 128, 128, 128, 128 };
static const s16 lap_lstm_wg_o[LSTM_HIDDEN_SIZE] = { 128, 128, 128, 128 };

static const s16 lap_lstm_wh_i[LSTM_HIDDEN_SIZE][LSTM_HIDDEN_SIZE] = {
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
};
static const s16 lap_lstm_wh_f[LSTM_HIDDEN_SIZE][LSTM_HIDDEN_SIZE] = {
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
};
static const s16 lap_lstm_wh_c[LSTM_HIDDEN_SIZE][LSTM_HIDDEN_SIZE] = {
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
};
static const s16 lap_lstm_wh_o[LSTM_HIDDEN_SIZE][LSTM_HIDDEN_SIZE] = {
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
    { 128, 128, 128, 128 },
};

static const s16 lap_lstm_b_i[LSTM_HIDDEN_SIZE] = { 0, 0, 0, 0 };
static const s16 lap_lstm_b_f[LSTM_HIDDEN_SIZE] = { 0, 0, 0, 0 };
static const s16 lap_lstm_b_c[LSTM_HIDDEN_SIZE] = { 0, 0, 0, 0 };
static const s16 lap_lstm_b_o[LSTM_HIDDEN_SIZE] = { 0, 0, 0, 0 };

static const s16 lap_lstm_fc_weight[LSTM_HIDDEN_SIZE] = { 128, 128, 128, 128 };
static const s16 lap_lstm_fc_bias = 0;

#define LAP_DEF_FREQ_STEP          5
#define LAP_MAX_FREQ_STEP_PERCENT  25
#define LAP_MIN_FREQ_STEP_PERCENT  5
#define LAP_DEF_SAMPLING_DOWN_FAC  2
#define LAP_MAX_SAMPLING_DOWN_FAC  5
#define LAP_DEF_SAMPLING_RATE      1

/* Function Prototypes */
static inline unsigned int lap_get_freq_step_khz(struct lap_tuners *tuners, struct cpufreq_policy *policy);
static unsigned int lap_dbs_get_load(struct cpufreq_policy *policy, bool ignore_nice);
static bool lap_is_on_ac(int *battery_capacity);
static s16 lap_lstm_scale_load(unsigned int load, unsigned int target);
static void lap_lstm_init(struct lap_lstm_state *state, s16 initial_sample);
static void lap_lstm_push(struct lap_lstm_state *state, s16 sample);
static s16 lap_lstm_predict(struct lap_lstm_state *state);
static void lap_apply_lstm_policy(struct cpufreq_policy *policy, struct lap_policy_info *lp, unsigned int load);
static unsigned long cs_dbs_update(struct cpufreq_policy *policy);
static void lap_work_handler(struct work_struct *work);

static inline s16 lap_lstm_clamp(s32 value)
{
    if (value > LSTM_MAX_VALUE)
        value = LSTM_MAX_VALUE;
    else if (value < LSTM_MIN_VALUE)
        value = LSTM_MIN_VALUE;
    return (s16)value;
}

static inline s16 lap_lstm_activate_sigmoid(s32 value)
{
    // Simplified sigmoid: 0.5 + 0.25 * x
    value = (LSTM_ONE / 2) + (value / 4);
    if (value > LSTM_ONE)
        return LSTM_ONE;
    if (value < 0)
        return 0;
    return (s16)value;
}

static inline s16 lap_lstm_activate_tanh(s32 value)
{
    // Simplified tanh: x / (1 + |x|)
    s32 abs_val = abs(value);
    value = div_s64((s64)value << LSTM_Q, LSTM_ONE + abs_val);
    return lap_lstm_clamp(value);
}

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

/* Detect efficiency and performance cores based on max frequency */
static void detect_clusters(struct cpufreq_policy *policy, struct cpumask *eff_mask, struct cpumask * lp_eff_mask, struct cpumask *perf_mask)
{
    unsigned int cpu;
    unsigned int max_freq, min_freq = UINT_MAX, highest_freq = 0;

    cpumask_clear(perf_mask);
    cpumask_clear(eff_mask);
    cpumask_clear(lp_eff_mask);

     /* Find min and max CPU frequencies */
    for_each_cpu(cpu, policy->cpus) {
        max_freq = cpufreq_quick_get_max(cpu);
        if (max_freq < min_freq) min_freq = max_freq;
        if (max_freq > highest_freq) highest_freq = max_freq;
    }

    /* Assign cores based on frequency tiers */
    for_each_cpu(cpu, policy->cpus) {
        max_freq = cpufreq_quick_get_max(cpu);
        if (max_freq == highest_freq)
            cpumask_set_cpu(cpu, perf_mask);
        else if (max_freq == min_freq)
            cpumask_set_cpu(cpu, lp_eff_mask);
        else
            cpumask_set_cpu(cpu, eff_mask);
    }

     pr_info("Detected clusters: %u Perf, %u E, %u LP-E\n",
            cpumask_weight(perf_mask),
            cpumask_weight(eff_mask),
            cpumask_weight(lp_eff_mask));
}

/* lap_dbs_get_load - compute average load with 3 cluster weights */
static unsigned int lap_dbs_get_load(struct cpufreq_policy *policy, bool ignore_nice)
{
    struct lap_policy_info *lp = policy->governor_data;
    unsigned int load_sum = 0, perf_load_sum = 0, e_load_sum = 0, lp_e_load_sum = 0;
    unsigned int cpu;
    unsigned int perf_cpus = 0, e_cpus = 0, lp_e_cpus = 0;
    u64 cur_time, cur_idle, cur_nice, idle_delta, nice_delta;
    unsigned int time_elapsed, cur_load;
    int battery_capacity;

    if (!lp || !cpumask_weight(policy->cpus))
        return 0;

    /* Detect clusters if not initialized */
    if (cpumask_empty(&lp->perf_mask) || cpumask_empty(&lp->eff_mask) || cpumask_empty(&lp->lp_eff_mask)) {
        detect_clusters(policy, &lp->eff_mask, &lp->lp_eff_mask, &lp->perf_mask);
    }

    perf_cpus = cpumask_weight(&lp->perf_mask);
    e_cpus = cpumask_weight(&lp->eff_mask);
    lp_e_cpus = cpumask_weight(&lp->lp_eff_mask);

    /* Compute load per CPU */
    for_each_cpu(cpu, policy->cpus) {
        struct lap_cpu_dbs *cdbs = per_cpu_ptr(&lap_cpu_dbs, cpu);
        cur_idle = get_cpu_idle_time_us(cpu, &cur_time);
        cur_nice = jiffies_to_usecs(kcpustat_cpu(cpu).cpustat[CPUTIME_NICE]);
        time_elapsed = (unsigned int)(cur_time - cdbs->prev_update_time);
        idle_delta = (unsigned int)(cur_idle - cdbs->prev_cpu_idle);
        nice_delta = (unsigned int)(cur_nice - cdbs->prev_cpu_nice);

        if (unlikely(time_elapsed == 0))
            cur_load = 100;
        else {
            unsigned int busy_time = time_elapsed - idle_delta;
            if (ignore_nice)
                busy_time -= nice_delta;
            cur_load = 100 * busy_time / time_elapsed;
        }

        cdbs->prev_cpu_idle = cur_idle;
        cdbs->prev_cpu_nice = cur_nice;
        cdbs->prev_update_time = cur_time;

        if (cpumask_test_cpu(cpu, &lp->perf_mask))
            perf_load_sum += cur_load;
        else if (cpumask_test_cpu(cpu, &lp->eff_mask))
            e_load_sum += cur_load;
        else if (cpumask_test_cpu(cpu, &lp->lp_eff_mask))
            lp_e_load_sum += cur_load;

        load_sum += cur_load;
    }

    /* Weighted load based on power source and battery */
    unsigned int final_load = 0;
    if (perf_cpus || e_cpus || lp_e_cpus) {
        unsigned int perf_load = perf_cpus ? perf_load_sum / perf_cpus : 0;
        unsigned int e_load = e_cpus ? e_load_sum / e_cpus : 0;
        unsigned int lp_e_load = lp_e_cpus ? lp_e_load_sum / lp_e_cpus : 0;

        if (!lap_is_on_ac(&battery_capacity) && battery_capacity <= 20)
            final_load = (lp_e_load * 7 + e_load * 2 + perf_load * 1) / 10;
        else if (!lap_is_on_ac(&battery_capacity))
            final_load = (lp_e_load * 5 + e_load * 3 + perf_load * 2) / 10;
        else
            final_load = (perf_load * 5 + e_load * 3 + lp_e_load * 2) / 10;

        return final_load;
    }

    return load_sum / cpumask_weight(policy->cpus);
}

/* lap_is_on_ac - Retrieves AC status and updates governor state */
static bool lap_is_on_ac(int *battery_capacity)
{
    struct power_supply *psy;
    union power_supply_propval val;
    int i;
    bool on_ac = false;

    *battery_capacity = 100;

    for (i = 0; i < ARRAY_SIZE(ac_names) && ac_names[i] != NULL; i++) {
        psy = power_supply_get_by_name(ac_names[i]);
        if (!psy)
            continue;
        
        if (power_supply_get_property(psy, POWER_SUPPLY_PROP_ONLINE, &val) == 0 && val.intval) {
            on_ac = true;
        }
        power_supply_put(psy);

        if (on_ac) {
            break;
        }
    }

    psy = power_supply_get_by_name("battery");
    if (psy) {
        if (power_supply_get_property(psy, POWER_SUPPLY_PROP_CAPACITY, &val) == 0) {
            *battery_capacity = val.intval;
        }
        power_supply_put(psy);
    }

    lap_on_ac_power = on_ac;
    return on_ac;
}

static s16 lap_lstm_scale_load(unsigned int load, unsigned int target)
{
    s64 diff = (s64)load - target;
    s64 scaled = diff * LSTM_ONE;

    scaled = div_s64(scaled, 100);
    scaled = clamp_t(s64, scaled, LSTM_MIN_VALUE, LSTM_MAX_VALUE);

    return (s16)scaled;
}

static void lap_lstm_init(struct lap_lstm_state *state, s16 initial_sample)
{
    int i;

    for (i = 0; i < LSTM_WINDOW; i++)
        state->history[i] = initial_sample;
    for (i = 0; i < LSTM_HIDDEN_SIZE; i++) {
        state->hidden[i] = 0;
        state->cell[i] = 0;
    }
}

static void lap_lstm_push(struct lap_lstm_state *state, s16 sample)
{
    memmove(&state->history[0], &state->history[1],
        (LSTM_WINDOW - 1) * sizeof(state->history[0]));
    state->history[LSTM_WINDOW - 1] = sample;
}

static s16 lap_lstm_predict(struct lap_lstm_state *state)
{
    s32 i, j;
    s32 acc;
    s16 input_gate[LSTM_HIDDEN_SIZE];
    s16 forget_gate[LSTM_HIDDEN_SIZE];
    s16 cell_gate[LSTM_HIDDEN_SIZE];
    s16 output_gate[LSTM_HIDDEN_SIZE];
    s16 sample;
    s32 fc_acc = 0;

    for (i = 0; i < LSTM_WINDOW; i++) {
        sample = state->history[i];

        for (j = 0; j < LSTM_HIDDEN_SIZE; j++) {
            // Input gate
            acc = (s32)lap_lstm_b_i[j] << LSTM_Q;
            acc += (s32)sample * lap_lstm_wg_i[j];
            acc += (s32)state->hidden[j] * lap_lstm_wh_i[j][j];
            input_gate[j] = lap_lstm_activate_sigmoid(acc >> LSTM_Q);

            // Forget gate
            acc = (s32)lap_lstm_b_f[j] << LSTM_Q;
            acc += (s32)sample * lap_lstm_wg_f[j];
            acc += (s32)state->hidden[j] * lap_lstm_wh_f[j][j];
            forget_gate[j] = lap_lstm_activate_sigmoid(acc >> LSTM_Q);

            // Cell gate
            acc = (s32)lap_lstm_b_c[j] << LSTM_Q;
            acc += (s32)sample * lap_lstm_wg_c[j];
            acc += (s32)state->hidden[j] * lap_lstm_wh_c[j][j];
            cell_gate[j] = lap_lstm_activate_tanh(acc >> LSTM_Q);

            // Update cell state
            state->cell[j] = ((s32)forget_gate[j] * state->cell[j] +
                            (s32)input_gate[j] * cell_gate[j]) >> LSTM_Q;

            // Output gate
            acc = (s32)lap_lstm_b_o[j] << LSTM_Q;
            acc += (s32)sample * lap_lstm_wg_o[j];
            acc += (s32)state->hidden[j] * lap_lstm_wh_o[j][j];
            output_gate[j] = lap_lstm_activate_sigmoid(acc >> LSTM_Q);

            // Update hidden state
            state->hidden[j] = ((s32)output_gate[j] *
                              lap_lstm_activate_tanh((s32)state->cell[j] << LSTM_Q)) >> LSTM_Q;
        }
    }

    for (i = 0; i < LSTM_HIDDEN_SIZE; i++) {
        fc_acc += (s32)state->hidden[i] * lap_lstm_fc_weight[i];
    }
    fc_acc = (fc_acc >> LSTM_Q) + lap_lstm_fc_bias;

    return lap_lstm_clamp(fc_acc);
}

static void lap_apply_lstm_policy(struct cpufreq_policy *policy,
                 struct lap_policy_info *lp, unsigned int load)
{
    unsigned int requested_freq = lp->requested_freq;
    unsigned int step_khz = lap_get_freq_step_khz(&lp->tuners, policy);
    s16 lstm_sample;
    s16 lstm_output;
    s64 scaled_delta;
    s64 delta_khz;

    lstm_sample = lap_lstm_scale_load(load, lp->tuners.target_load);
    lap_lstm_push(&lp->lstm, lstm_sample);

    if (load >= LAP_HIGH_LOAD_BYPASS) {
        requested_freq = policy->max;
    } else if (load <= LAP_LOW_LOAD_BYPASS) {
        requested_freq = policy->min;
    } else {
        lstm_output = lap_lstm_predict(&lp->lstm);
        lstm_output = clamp_t(s16, lstm_output, -LSTM_ONE, LSTM_ONE);

        scaled_delta = (s64)lstm_output * lp->tuners.learning_rate_fp;
        delta_khz = (scaled_delta * step_khz) >> (LSTM_Q + FP_SHIFT);
        delta_khz = clamp_t(s64, delta_khz, -(s64)step_khz, (s64)step_khz);

        requested_freq = clamp_val((s64)requested_freq + delta_khz,
                       policy->min, policy->max);
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

    if (!lp)
        return HZ;

    tuners = &lp->tuners;
    mutex_lock(&lp->lock);

    if (policy->cpu == 0) {
        int battery_capacity;
        lap_on_ac_power = lap_is_on_ac(&battery_capacity);
    }

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
    lap_global_load = lap_dbs_get_load(policy, tuners->ignore_nice_load);
    mutex_unlock(&lap_global_lock);

    load = lap_global_load;

    if (lp->last_target_load != tuners->target_load) {
        s16 reset_sample = lap_lstm_scale_load(load, tuners->target_load);
        lap_lstm_init(&lp->lstm, reset_sample);
        lp->last_target_load = tuners->target_load;
    }

    lap_apply_lstm_policy(policy, lp, load);

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

        mutex_lock(&lap_global_lock);
        current_load = lap_global_load;
        mutex_unlock(&lap_global_lock);

        lap_lstm_init(&lp->lstm, lap_lstm_scale_load(current_load, val));
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
    .name = "baram"
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
    lap_lstm_init(&lp->lstm, 0);
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
    .name = "baram",
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

MODULE_DESCRIPTION("'cpufreq_baram' - Conservative-style governor for laptops with lightweight LSTM inference");

MODULE_LICENSE("GPL");



module_init(baram_module_init);

module_exit(baram_module_exit);
