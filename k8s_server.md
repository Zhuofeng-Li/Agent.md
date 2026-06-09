# Slime Development Guide

## Mandatory Rules

1. **Activate venv** before any Python/uv command on dev machine:
   ```bash
   source /local/home/zhuofeng/slime/.venv/bin/activate
   ```

2. **AGENTS.md**: Before running/modifying any script, check for `AGENTS.md` in the same directory and follow it. Mandatory for `scripts/frontier_cs/`.

3. **All code changes must be made on dev machine only**, then synced to compute node. Never modify code directly on the compute node.

4. **All scripts must run on compute node**, never dev machine. Default node: `xliucr-slime-16n-june3-worker-0`.

5. **Always sync before running anything on compute node** — dev machine may have newer code.

## Environment Paths

| Location | Path |
|---|---|
| Dev machine | `/local/home/zhuofeng/slime/` |
| S3 staging | `s3://p11-dev/zhuofeng/slime/` |
| Compute node | `/shared/dev/zhuofeng/slime/` |
| s5cmd (dev) | `/local/home/zhuofeng/slime/.venv/bin/s5cmd` |
| s5cmd (compute) | `/shared/crenyuan/s5cmd` |

> 旧 S3 路径 `s3://rufus-post-training-users-272436634516-us-west-2-an/...` 计算节点无权限，不要用。

## Sync Sequence（每次必须执行）

```bash
# 1. Dev machine → S3
/local/home/zhuofeng/slime/.venv/bin/s5cmd sync \
  --exclude ".git/*" --exclude ".venv/*" --exclude "__pycache__/*" \
  --exclude "*.pyc" --exclude ".pytest_cache/*" \
  /local/home/zhuofeng/slime/ s3://p11-dev/zhuofeng/slime/

# 2. kubectl exec（去掉 -it，否则报 Unable to use a TTY）
kubectl exec xliucr-slime-16n-june3-worker-0 -n application-nonprod -- bash -c "
  cd /shared/dev/zhuofeng
  /shared/crenyuan/s5cmd sync 's3://p11-dev/zhuofeng/slime/*' /shared/dev/zhuofeng/slime/
  cd slime && bash scripts/<script-name>.sh
"
```

> After `kubectl exec`, always `cd /shared/dev/zhuofeng` — default dir is `/shared/dev/xliucr`.

## Compute Node 注意事项

- 没有 `sudo`，没有 `aws` CLI
- Slime 训练不能用 Docker；Judge 服务可以用 Docker，但路径必须在 `/shared/dev/zhuofeng` 下
- 安装依赖用 `pip install`（用 Docker 装依赖会导致所有 8 节点被 kill）
- 节点 IP 用 `kubectl exec <pod> -- hostname -I` 获取，不要假设固定

## June3 Cluster 节点 IP（当前默认集群：xliucr-slime-16n-june3，16 节点 × 8 H200 = 128 GPU）

默认计算节点：`xliucr-slime-16n-june3-worker-0`

```
10.209.123.106  # worker-0 (master)
10.209.199.180  # worker-1
10.209.99.1     # worker-2
10.209.60.130   # worker-3
10.209.80.130   # worker-4
10.209.142.137  # worker-5
10.209.36.22    # worker-6
10.209.93.243   # worker-7
10.209.117.165  # worker-8
10.209.115.76   # worker-9
10.209.228.200  # worker-10
10.209.171.185  # worker-11
10.209.56.67    # worker-12
10.209.30.222   # worker-13
10.209.216.87   # worker-14
10.209.31.84    # worker-15
```

## 多节点训练（GLM-4.7-355B-A32B，8 节点 64 GPU）

脚本：`scripts/run-glm4.7-355B-A32B.sh`，必须提前设置：

| 变量 | 说明 |
|---|---|
| `MASTER_ADDR` | worker-0 IP（`hostname -I \| awk '{print $1}'`） |
| `HOSTFILE` | 8 个节点 IP 文件，每行一个 |
| `BASE_DIR` | 模型/数据根目录（`/shared/dev/zhuofeng/slime`） |

pod 间无 sshd，多节点 Ray 用 `kubectl exec` 逐节点启动，见 `scripts/launch-glm4.7-355B-multinode.sh`。

多节点 torchrun 从 dev machine 触发（pod 内无 kubectl）：
```bash
kubectl exec worker-1 -n ns -- bash -c "nohup torchrun ... --node-rank 1 ... &"
sleep 3
kubectl exec worker-0 -n ns -- bash -c "nohup torchrun ... --node-rank 0 ... &"
```

## GLM-4.7-355B 完整启动流程

1. **下载模型**（计算节点，后台，~700G，1-2 小时）
2. **下载数据集**：`dapo-math-17k` → `$BASE_DIR/dapo-math-17k/`，`aime-2024` → `$BASE_DIR/rl_data/`
3. **转换 checkpoint**：`scripts/convert-glm4.7-355B-torch-dist.sh`（2 节点 × 8 GPU）
4. **启动训练**：`scripts/launch-glm4.7-355B-multinode.sh`（在 worker-0 运行）

## 经验坑点

### HuggingFace 下载
- 正确 repo ID：`zhuzilin/dapo-math-17k`、`zhuzilin/aime-2024`（`Bytedance-Seed/DAPO-Math-17k` 会 404）
- HF API 3000 req/5min 限制，并发 snapshot_download 易触发 429，等 ~200s 重试
- 用 `HF_XET_HIGH_PERFORMANCE=1`（新版，替代已废弃的 `HF_HUB_ENABLE_HF_TRANSFER=1`）
- 绕过 rate limit 直接 wget：
  ```bash
  wget -q --header="Authorization: Bearer $HF_TOKEN" \
    'https://huggingface.co/datasets/zhuzilin/dapo-math-17k/resolve/main/dapo-math-17k.jsonl' \
    -O $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
  ```

### bash 数组在子 shell 不展开
`MODEL_ARGS` 数组在 `nohup bash -c '${MODEL_ARGS[@]}'` 里不展开，导致参数丢失。在 dev machine 上提前展开成字符串：
```bash
MODEL_ARGS_STR=$(bash -c "source scripts/models/glm4.5-355B-A32B.sh; echo \"\${MODEL_ARGS[*]}\"")
```

### logs 目录需提前创建
ray job submit 里有日志重定向时，目录必须存在：
```bash
mkdir -p /shared/dev/zhuofeng/slime/logs
```
