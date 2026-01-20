""" TPU VM 作成からコマンド実行、削除までの自動化 """
from sys import stderr
from random import choice
from subprocess import run, CalledProcessError

from absl import app, flags

flags.DEFINE_string('queue_name', "exp", "TPU Queued Resource の名前")
flags.DEFINE_multi_string('zone', "us-west4-a", "TPU を作るzone、複数指定するとランダムに選ばれる")
flags.DEFINE_string('node_id', None, "TPU node の名前")
flags.DEFINE_string('type', "v5litepod-16", "TPU の種類")
flags.DEFINE_boolean('spot', True, "Spot VM にする")
flags.DEFINE_string('runtime', "v2-alpha-tpuv5-lite", "TPU VM のランタイムバージョン")
flags.DEFINE_string('tfversion', "==2.19.1", "TPU VMにインストールするtensorflowのバージョン指定")
flags.DEFINE_multi_string('scpfile', None, "TPU VM にコピーするファイル(複数可)")
flags.DEFINE_string('runcmd', None, "TPU VM で実行するコマンド")
flags.DEFINE_boolean('single_worker', False, "ssh, scp をひとつのワーカのみに対して実行する")

FLAGS = flags.FLAGS

def run_command(*args, xtrace=True, key=None):
    xtrace and print("+", *args, file=stderr, flush=True)
    if not key:
        run(args, check=True)
    else:
        return run([*args, f"--format=get({key})"], check=True, capture_output=True).stdout.decode().strip()

def qrcmd_cmn(cmd):
    return "gcloud", "compute", "tpus", "queued-resources", cmd

def qrcmd(cmd):
    return *qrcmd_cmn(cmd), FLAGS.queue_name, f"--zone={tpu_zone}"

def qrscp(files):
    cmd = [*qrcmd_cmn("scp"), *files, f"{FLAGS.queue_name}:", f"--zone={tpu_zone}"]
    FLAGS.single_worker or cmd.append("--worker=all")
    return cmd

def qrssh(command):
    cmd = [*qrcmd("ssh"), f"--command={command}"]
    FLAGS.single_worker or cmd.append("--worker=all")
    return cmd

def main(argv):
    global tpu_zone
    tpu_zone = choice(FLAGS.zone)
    if not FLAGS.node_id: FLAGS.node_id = FLAGS.queue_name
    try:
        cmd = [*qrcmd("create"), f"--node-id={FLAGS.node_id}"]
        cmd += [f"--accelerator-type={FLAGS.type}", f"--runtime-version={FLAGS.runtime}"]
        FLAGS.spot and cmd.append("--spot")
        run_command(*cmd)
        stat = run_command(*qrcmd("describe"), key="state.state")
        while stat in ("WAITING_FOR_RESOURCES", "PROVISIONING"):
            stat = run_command(*qrcmd("describe"), xtrace=False, key="state.state")
        if stat != "ACTIVE": raise RuntimeError(f"unexpected queued resource state: {stat}")
        run_command(*qrssh("sudo pip3 install --progress-bar off --upgrade pip"))
        pip_install = "sudo pip3 install --progress-bar off --root-user-action ignore"
        run_command(*qrssh(f"{pip_install} tensorflow-tpu{FLAGS.tfversion}"))
        if FLAGS.scpfile: run_command(*qrscp(FLAGS.scpfile))
        run_command(*qrssh(FLAGS.runcmd))
    finally:
        try: run_command(*qrcmd("describe"), xtrace=False, key="state.state")
        except CalledProcessError: pass
        else: run_command(*qrcmd("delete"), "--force", "--quiet")

if __name__ == '__main__':
    flags.mark_flag_as_required('runcmd')
    app.run(main)
