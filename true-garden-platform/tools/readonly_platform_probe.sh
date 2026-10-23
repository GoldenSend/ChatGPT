#!/usr/bin/env bash
# Read-only platform probe for PS4 #1 True Garden (Track B, R1).
#
# Run on the IdeaPad, only after the GPU thread has released the PS4:
#   bash readonly_platform_probe.sh > probe-$(date -u +%Y%m%dT%H%M%SZ).txt
#
# What it does: one SSH session to ps4-garden that only reads /proc, /sys,
# the kernel log, the journal and package metadata. It writes nothing on the
# PS4, opens no block device, loads or binds no driver, and never touches
# /dev/dri. It never reads /key or the initramfs (the loader puts the
# console's HDD key there). It does not run blkid, smartctl or hdparm.
#
# Probe IDs (P1..P10) match the UNKNOWN rows in SUPPORT-MATRIX.md.
# WITH_GPU_SYSFS=1 adds two amdgpu sysfs/debugfs queries (P6). Leave it off
# while any GPU job may be running.
set -euo pipefail

HOST="${HOST:-ps4-garden}"
WITH_GPU_SYSFS="${WITH_GPU_SYSFS:-0}"

ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "WITH_GPU_SYSFS=$WITH_GPU_SYSFS bash -s" <<'REMOTE'
set -u
t() { timeout 10 "$@" 2>&1 || echo "(exit $?)"; }
h() { printf '\n=== %s\n' "$*"; }

h P0 identity
t date -u +%FT%TZ
t hostname
t uname -r
t cat /proc/sys/kernel/random/boot_id
t cat /proc/uptime
t findmnt -n -o SOURCE,UUID,TARGET,FSTYPE,OPTIONS /
t systemctl is-active atom-worker.service ssh.service ps4-sky2-carrier-guard.service

h P1 southbridge revision and PCI functions
t lspci -nnk -s 00:14
t sh -c 'dmesg 2>/dev/null | grep -i -E "aeolia|belize|baikal|southbridge|apcie" | head -40'

h P2 storage: block devices, root, ATA errors
t lsblk -d -o NAME,TRAN,SERIAL,MODEL,SIZE,RO,HOTPLUG
t sh -c 'dmesg 2>/dev/null | grep -E "ata[0-9]+(\.[0-9]+)?:" | head -60'
t sh -c 'for f in /sys/fs/ext4/*/errors_count; do echo "$f $(cat $f)"; done'
t sh -c 'grep -H . /sys/class/scsi_host/host*/proc_name 2>/dev/null'

h P3 mmc0 identity and errors, wireless
t sh -c 'for d in /sys/class/mmc_host/*; do echo "$d -> $(readlink -f $d/device)"; done'
t sh -c 'for d in /sys/bus/mmc/devices/*; do echo "$d"; cat $d/uevent 2>/dev/null; done'
t sh -c 'dmesg 2>/dev/null | grep -i -E "mmc[0-9]|sdhci|mt76|mt7668|mwifiex|btmtk|bluetooth|hci0" | head -60'
t sh -c 'ls /sys/class/net; ls /sys/class/bluetooth 2>/dev/null'

h D-state census with wait channel, 3 samples 2 s apart
for i in 1 2 3; do
  date -u +%T
  ps -eo pid,ppid,stat,etime,wchan:32,comm --no-headers | awk '$3 ~ /^D/'
  sleep 2
done

h P4 hwmon and fan drivers
t sh -c 'for d in /sys/class/hwmon/hwmon*; do echo "$d name=$(cat $d/name 2>/dev/null)"; ls $d | grep -E "^(temp|fan|pwm)" ; done'
t sh -c 'grep -H . /sys/class/hwmon/hwmon*/temp*_input /sys/class/hwmon/hwmon*/fan*_input 2>/dev/null'
t sh -c 'ls /sys/class/thermal 2>/dev/null; grep -H . /sys/class/thermal/thermal_zone*/type 2>/dev/null'

h P5 CPU and frequency policy
t sh -c 'grep -m1 "model name" /proc/cpuinfo; grep -c ^processor /proc/cpuinfo; grep -m1 ^flags /proc/cpuinfo'
t sh -c 'ls /sys/devices/system/cpu/cpufreq 2>/dev/null || echo "no cpufreq"'
t sh -c 'grep -H . /sys/devices/system/cpu/cpu0/cpufreq/scaling_* 2>/dev/null'

h P6 GPU driver state without touching /dev/dri
t cat /proc/cmdline
t sh -c 'dmesg 2>/dev/null | grep -i -E "amdgpu|\[drm\]|liverpool|sclk" | head -60'
t sh -c 'grep -H . /sys/module/amdgpu/parameters/{dpm,vramlimit,gartsize,gttsize,ppfeaturemask} 2>/dev/null'
if [ "${WITH_GPU_SYSFS:-0}" = 1 ]; then
  t sh -c 'cat /sys/class/drm/card0/device/pp_dpm_sclk 2>&1 | head'
  t sh -c 'cat /sys/kernel/debug/dri/0/amdgpu_pm_info 2>&1 | head -30'
fi

h P7 display connectors
t sh -c 'for c in /sys/class/drm/card*-*; do echo "$c $(cat $c/status 2>/dev/null)"; done'

h P8 memory, swap, zram, OOM
t free -m
t cat /proc/swaps
t sh -c 'ls /sys/block | grep -E "^zram" || echo "no zram device"'
t grep -E "^oom_kill " /proc/vmstat

h P9 kernel config: containers, KVM, filesystems, USB NICs
if [ -r /proc/config.gz ]; then
  zcat /proc/config.gz | grep -E "^CONFIG_(NAMESPACES|USER_NS|PID_NS|NET_NS|CGROUPS|CGROUP_BPF|MEMCG|OVERLAY_FS|BTRFS_FS|XFS_FS|F2FS_FS|EXFAT_FS|ZRAM|ZSWAP|KVM|KVM_AMD|USB_RTL8152|USB_NET_AX88179_178A|SECCOMP|BPF_SYSCALL|IO_URING|FW_LOADER_COMPRESS)=" || true
elif [ -r "/boot/config-$(uname -r)" ]; then
  grep -E "^CONFIG_(NAMESPACES|USER_NS|CGROUPS|MEMCG|OVERLAY_FS|ZRAM|KVM_AMD|USB_RTL8152|IO_URING)=" "/boot/config-$(uname -r)" || true
else
  echo "no kernel config exposed"
fi
t sh -c 'cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null || echo "no cgroup v2 root"'
t sh -c 'ls /dev/kvm 2>/dev/null || echo "no /dev/kvm"'

h P10 platform devices: watchdog, LED, power button, input
t sh -c 'ls /dev/watchdog* 2>/dev/null || echo "no watchdog device"'
t sh -c 'ls /sys/class/leds 2>/dev/null; cat /proc/bus/input/devices 2>/dev/null | grep -E "^N:"'

h "Userspace graphics packages, metadata only"
t sh -c 'dpkg-query -W -f="\${Package} \${Version}\n" "libdrm*" "mesa*" "libgl1-mesa*" "libegl-mesa*" 2>/dev/null | head -20'

h "Recent kernel warnings, last 80 at warn or worse"
t sh -c 'dmesg --level=emerg,alert,crit,err,warn 2>/dev/null | tail -80'
REMOTE
