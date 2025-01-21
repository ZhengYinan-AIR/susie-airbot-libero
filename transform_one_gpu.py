import jax
import orbax.checkpoint

# 1. 恢复模型
checkpoint_path = "/home/user/zyn/susie-airbot-libero/logs/2025.01.21_00.10.32/40000/params_ema"
agent = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_path, item=None)


device = jax.devices()[0]  # 假设选择 GPU 0，或者可以选择 CPU

# 3. 将模型的所有参数放到目标设备上（将多卡模型迁移到单卡设备）
def move_model_to_device(model, device):
    # 使用 jax.device_put 将每个参数放到目标设备
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, device), model)

# 将模型移到单卡
agent_on_single_device = move_model_to_device(agent, device)



# 4. 重新保存模型到 checkpoint
new_checkpoint_path = "/home/user/zyn/susie-airbot-libero/logs/airbot/40000"
orbax.checkpoint.PyTreeCheckpointer().save(new_checkpoint_path, agent_on_single_device)

print("模型已经成功保存为单卡版本，路径：", new_checkpoint_path)
