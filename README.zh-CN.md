# cVAE API

## 项目概述
cVAE API 是一个基于 Flask 的服务，用于暴露已训练好的条件变分自编码器（conditional Variational Autoencoder，cVAE），帮助研究者依据行星整体观测量预测岩石系外行星的内部结构概率分布。应用在启动时会加载预训练的 PyTorch 模型和特征缩放器，支持以 JSON 或文件上传的方式进行推理。

## 主要特性
- 在应用启动时加载预训练的 cVAE 模型与缩放器，确保推理结果可复现。
- 支持标量 JSON 输入、批量 JSON 输入以及 `.npy`、`.csv`、`.xlsx`、`.parquet` 等文件格式的预测请求。
- 当未提供 `Times` 参数时会自动使用默认值 10，表示为每条输入生成 10 组样本。
- 输出包含八个行星内部结构相关参数，并对压力和温度等量纲进行了单位还原。

## 目录结构
```
.
├── app/
│   ├── __init__.py        # Flask 应用工厂及模型加载
│   ├── routes.py          # REST API 路由与推理逻辑
│   ├── config.py          # 配置常量
│   └── cvae.py            # cVAE 模型结构定义
├── static/                # 预训练权重与缩放器
├── requirements.txt       # 运行依赖
└── gunicorn_conf.py       # Gunicorn 配置
```

## 环境要求
- Python 3.10 及以上
- `requirements.txt` 中列出的依赖（Flask、PyTorch、pandas 等）。
- 将预训练资源放置于 `static/best_model.pth`、`static/Xscaler.save`、`static/yscaler.save`。
- 至少 1 GB 可用内存，用于加载预训练模型和缩放器。

## 本地运行
1. 创建并激活虚拟环境。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 设置 Flask 入口并启动开发服务器（监听 `8000` 端口）：
   ```bash
   export FLASK_APP=app:create_app
   flask run --debug --port 8000
   ```
   默认监听 `http://127.0.0.1:8000/api/`。

若需使用 Gunicorn（生产环境推荐）：
```bash
gunicorn --config gunicorn_conf.py 'app:create_app()'
```

## 接口说明
所有接口均以 `/api` 为前缀。

### `GET /api/hello`
健康检查接口，返回简单的问候字符串。

```bash
curl http://127.0.0.1:8000/api/hello
```

### `POST /api/single_prediction`
请求体为 JSON，需提供四个必需特征（`Mass`、`Radius`、`Fe/Mg`、`Si/Mg`），每个字段建议使用列表形式以支持批量推理，可选字段 `Times` 控制每条输入生成的样本数量（默认 10）。示例：
```json
{
  "Mass": [1.0],
  "Radius": [1.0],
  "Fe/Mg": [0.5],
  "Si/Mg": [1.0],
  "Times": 25
}
```
响应会包含原始请求和按批次索引划分的 `Prediction_distribution` 结果。

```bash
curl -X POST http://127.0.0.1:8000/api/single_prediction \
  -H "Content-Type: application/json" \
  -d '{
        "Mass": [1.0],
        "Radius": [1.0],
        "Fe/Mg": [0.5],
        "Si/Mg": [1.0],
        "Times": 25
      }'
```

### `POST /api/multi_prediction`
与 `single_prediction` 接口的请求格式相同，主要用于同时提交多组行星数据。

### `POST /api/file_prediction`
接收 multipart form-data：
- `file`：上传包含四个特征列（顺序为 `[Mass, Radius, Fe/Mg, Si/Mg]`）的 `.npy`、`.csv`、`.xlsx` 或 `.parquet` 文件。
- 可选 `Times` 表单字段（默认 10）。

接口会返回每行输入对应的预测分布，质量和半径相关的输出会在返回前转换回原始单位。

## 模型输出参数
- `WRF` —— Water Radial Fraction（水层径向占比）
- `MRF` —— Mantle Radial Fraction（地幔径向占比）
- `CRF` —— Core Radial Fraction（地核径向占比）
- `WMF` —— Water Mass Fraction（水层质量占比）
- `CMF` —— Core Mass Fraction（地核质量占比）
- `P_CMB (TPa)` —— 核幔边界压力（TeraPascal）
- `T_CMB (10^3K)` —— 核幔边界温度（以千开尔文为单位）
- `K2` —— 潮汐 Love 数

## 贡献指南
欢迎提交 Pull Request。请确保代码格式规范，并为新功能补充相应的文档说明。

## 许可协议
如有许可证，请在此处说明。
