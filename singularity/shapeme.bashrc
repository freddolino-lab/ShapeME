source /opt/conda/etc/profile.d/conda.sh
conda activate motifer

#export PATH="${PATH}:${PARDIR}"
#export PYTHONPATH="/conda/envs/ipod_p3/lib/python3.zip:/conda/envs/ipod_p3/lib/python3:/conda/envs/ipod_p3/lib/python3/plat-linux2:/conda/envs/ipod_p3/lib/python3/lib-tk:/conda/envs/ipod_p3/lib/python3/lib-old:/conda/envs/ipod_p3/lib/python3/lib-dynload:/conda/envs/ipod_p3/lib/python3/site-packages"

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
  # We have color support; assume it's compliant with Ecma-48
  # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
  # a case would tend to support setf rather than setaf.)
  color_prompt=yes
    else
  color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt
