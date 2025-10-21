obj-m += cpufreq_baram.o

KVERSION = $(shell uname -r)

KDIR = /lib/modules/$(KVERSION)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
