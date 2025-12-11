from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from PIL import Image
import io
import json
from typing import List, Dict
import base64
import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.tester import Tester
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from torchvision import transforms
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class Config:
    """配置参数类，对应parse_args()函数的所有参数"""
    # Data input settings
    image_dir: str = 'data/iu_xray/images/'
    ann_path: str = 'data/iu_xray/annotation.json'
    
    # Data loader settings
    dataset_name: str = 'iu_xray'
    max_seq_length: int = 60
    threshold: int = 3
    num_workers: int = 2
    batch_size: int = 16
    
    # Model settings (for visual extractor)
    visual_extractor: str = 'resnet101'
    visual_extractor_pretrained: bool = True
    
    # Model settings (for Transformer)
    d_model: int = 512
    d_ff: int = 512
    d_vf: int = 2048
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    logit_layers: int = 1
    bos_idx: int = 0
    eos_idx: int = 0
    pad_idx: int = 0
    use_bn: int = 0
    drop_prob_lm: float = 0.5
    
    # for Relational Memory
    rm_num_slots: int = 3
    rm_num_heads: int = 8
    rm_d_model: int = 512
    
    # Sample related
    sample_method: str = 'beam_search'
    beam_size: int = 3
    temperature: float = 1.0
    sample_n: int = 1
    group_size: int = 1
    output_logsoftmax: int = 1
    decoding_constraint: int = 0
    block_trigrams: int = 1
    
    # Trainer settings
    n_gpu: int = 1
    epochs: int = 100
    save_dir: str = 'results/iu_xray'
    record_dir: str = 'records/'
    save_period: int = 1
    monitor_mode: str = 'max'
    monitor_metric: str = 'BLEU_4'
    early_stop: int = 50
    
    # Optimization
    optim: str = 'Adam'
    lr_ve: float = 5e-5
    lr_ed: float = 1e-4
    weight_decay: float = 5e-5
    amsgrad: bool = True
    
    # Learning Rate Scheduler
    lr_scheduler: str = 'StepLR'
    step_size: int = 50
    gamma: float = 0.1
    
    # Others
    seed: int = 9233
    resume: Optional[str] = None
    load: Optional[str] = "data/model_iu_xray.pth"

args = Config()
a = torch.load("data/model_iu_xray.pth")
tokenizer = Tokenizer(args)
model = R2GenModel(args, tokenizer)
model.load_state_dict(a['state_dict'])
model.to(torch.device('cuda:0'))


app = FastAPI(title="图像处理网站", description="上传两张图像并查看处理结果")

# 启用CORS以允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建上传目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """返回前端HTML页面"""
    # 这里我们将从外部文件读取HTML，但为了简单起见，我们返回一个消息
    # 在实际部署中，应该将前端HTML文件放在static目录中
    return HTMLResponse(content="请访问 /frontend 来查看前端页面")

@app.get("/frontend", response_class=HTMLResponse)
async def serve_frontend():
    """提供前端页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>图像处理网站</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            h1 {
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 2.5rem;
            }
            
            .subtitle {
                color: #7f8c8d;
                font-size: 1.1rem;
            }
            
            .content-area {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-template-rows: auto 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }
            
            @media (max-width: 900px) {
                .content-area {
                    grid-template-columns: 1fr;
                }
            }
            
            .upload-box {
                background: white;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .upload-box:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
            }
            
            .upload-box h2 {
                color: #3498db;
                margin-bottom: 15px;
                font-size: 1.8rem;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .upload-box h2 i {
                color: #2980b9;
            }
            
            .upload-area {
                border: 3px dashed #3498db;
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                cursor: pointer;
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                transition: all 0.3s ease;
                margin-bottom: 20px;
            }
            
            .upload-area:hover {
                background-color: #f8fafc;
                border-color: #2980b9;
            }
            
            .upload-area i {
                font-size: 4rem;
                color: #3498db;
                margin-bottom: 15px;
            }
            
            .upload-area p {
                color: #7f8c8d;
                font-size: 1.1rem;
                margin-bottom: 10px;
            }
            
            .file-info {
                font-size: 0.9rem;
                color: #95a5a6;
            }
            
            .image-preview {
                margin-top: 15px;
                max-height: 200px;
                border-radius: 5px;
                overflow: hidden;
                display: none;
            }
            
            .image-preview img {
                width: 100%;
                height: auto;
                object-fit: contain;
            }
            
            .output-box {
                background: white;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                grid-column: span 2;
                display: flex;
                flex-direction: column;
            }
            
            @media (max-width: 900px) {
                .output-box {
                    grid-column: span 1;
                }
            }
            
            .output-box h2 {
                color: #2ecc71;
                margin-bottom: 20px;
                font-size: 1.8rem;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .output-box h2 i {
                color: #27ae60;
            }
            
            .output-content {
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                padding: 20px;
                flex-grow: 1;
                min-height: 200px;
                background-color: #f9f9f9;
                overflow-y: auto;
            }
            
            .empty-output {
                color: #95a5a6;
                font-style: italic;
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100%;
            }
            
            .empty-output i {
                font-size: 3rem;
                margin-bottom: 15px;
                color: #bdc3c7;
            }
            
            .processing-result {
                display: none;
            }
            
            .result-images {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 25px;
            }
            
            .result-image {
                flex: 1;
                min-width: 300px;
            }
            
            .result-image img {
                width: 100%;
                border-radius: 5px;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
            }
            
            .result-details {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            
            .result-details h3 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            
            .result-details p {
                margin-bottom: 8px;
                color: #34495e;
            }
            
            .controls {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-top: 20px;
            }
            
            button {
                padding: 14px 30px;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .process-btn {
                background: #3498db;
                color: white;
            }
            
            .process-btn:hover {
                background: #2980b9;
                transform: translateY(-2px);
                box-shadow: 0 5px 10px rgba(52, 152, 219, 0.3);
            }
            
            .reset-btn {
                background: #e74c3c;
                color: white;
            }
            
            .reset-btn:hover {
                background: #c0392b;
                transform: translateY(-2px);
                box-shadow: 0 5px 10px rgba(231, 76, 60, 0.3);
            }
            
            button:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
                transform: none !important;
                box-shadow: none !important;
            }
            
            footer {
                text-align: center;
                margin-top: 30px;
                color: #7f8c8d;
                font-size: 0.9rem;
                padding: 15px;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .loading-spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1><i class="fas fa-images"></i> 图像处理与展示网站</h1>
                <p class="subtitle">上传两张图像，系统将进行处理并显示结果信息</p>
            </header>
            
            <div class="content-area">
                <div class="upload-box">
                    <h2><i class="fas fa-upload"></i> 图像上传区域 1</h2>
                    <div class="upload-area" id="uploadArea1">
                        <i class="fas fa-cloud-upload-alt"></i>
                        <p>点击或拖拽图像到此处上传</p>
                        <p class="file-info">支持 JPG, PNG, GIF 格式，最大 5MB</p>
                    </div>
                    <input type="file" id="fileInput1" accept="image/*" style="display: none;">
                    <div class="image-preview" id="preview1"></div>
                </div>
                
                <div class="upload-box">
                    <h2><i class="fas fa-upload"></i> 图像上传区域 2</h2>
                    <div class="upload-area" id="uploadArea2">
                        <i class="fas fa-cloud-upload-alt"></i>
                        <p>点击或拖拽图像到此处上传</p>
                        <p class="file-info">支持 JPG, PNG, GIF 格式，最大 5MB</p>
                    </div>
                    <input type="file" id="fileInput2" accept="image/*" style="display: none;">
                    <div class="image-preview" id="preview2"></div>
                </div>
                
                <div class="output-box">
                    <h2><i class="fas fa-chart-bar"></i> 处理结果输出区域</h2>
                    <div class="output-content" id="outputContent">
                        <div class="empty-output" id="emptyOutput">
                            <i class="fas fa-info-circle"></i>
                            <p>上传两张图像后，点击"处理图像"按钮查看结果</p>
                        </div>
                        <div class="processing-result" id="processingResult">
                            <div class="result-images" id="resultImages"></div>
                            <div class="result-details" id="resultDetails"></div>
                        </div>
                        <div class="loading" id="loading">
                            <div class="loading-spinner"></div>
                            <p>正在处理图像，请稍候...</p>
                        </div>
                    </div>
                    
                    <div class="controls">
                        <button class="process-btn" id="processBtn" disabled>
                            <i class="fas fa-cogs"></i> 处理图像
                        </button>
                        <button class="reset-btn" id="resetBtn">
                            <i class="fas fa-redo"></i> 重置所有
                        </button>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>© 2023 图像处理网站 | 使用 FastAPI 和 HTML/CSS/JavaScript 构建</p>
            </footer>
        </div>
        
        <script>
            // 全局变量
            let image1 = null;
            let image2 = null;
            const maxFileSize = 5 * 1024 * 1024; // 5MB
            
            // DOM 元素
            const uploadArea1 = document.getElementById('uploadArea1');
            const uploadArea2 = document.getElementById('uploadArea2');
            const fileInput1 = document.getElementById('fileInput1');
            const fileInput2 = document.getElementById('fileInput2');
            const preview1 = document.getElementById('preview1');
            const preview2 = document.getElementById('preview2');
            const processBtn = document.getElementById('processBtn');
            const resetBtn = document.getElementById('resetBtn');
            const emptyOutput = document.getElementById('emptyOutput');
            const processingResult = document.getElementById('processingResult');
            const loading = document.getElementById('loading');
            const resultImages = document.getElementById('resultImages');
            const resultDetails = document.getElementById('resultDetails');
            
            // 初始化
            document.addEventListener('DOMContentLoaded', () => {
                setupEventListeners();
            });
            
            // 设置事件监听器
            function setupEventListeners() {
                // 上传区域1
                uploadArea1.addEventListener('click', () => fileInput1.click());
                fileInput1.addEventListener('change', (e) => handleFileSelect(e, 1));
                uploadArea1.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea1.style.borderColor = '#2980b9';
                    uploadArea1.style.backgroundColor = '#f0f8ff';
                });
                uploadArea1.addEventListener('dragleave', () => {
                    uploadArea1.style.borderColor = '#3498db';
                    uploadArea1.style.backgroundColor = '';
                });
                uploadArea1.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea1.style.borderColor = '#3498db';
                    uploadArea1.style.backgroundColor = '';
                    if (e.dataTransfer.files.length) {
                        handleFileSelect({target: {files: e.dataTransfer.files}}, 1);
                    }
                });
                
                // 上传区域2
                uploadArea2.addEventListener('click', () => fileInput2.click());
                fileInput2.addEventListener('change', (e) => handleFileSelect(e, 2));
                uploadArea2.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea2.style.borderColor = '#2980b9';
                    uploadArea2.style.backgroundColor = '#f0f8ff';
                });
                uploadArea2.addEventListener('dragleave', () => {
                    uploadArea2.style.borderColor = '#3498db';
                    uploadArea2.style.backgroundColor = '';
                });
                uploadArea2.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea2.style.borderColor = '#3498db';
                    uploadArea2.style.backgroundColor = '';
                    if (e.dataTransfer.files.length) {
                        handleFileSelect({target: {files: e.dataTransfer.files}}, 2);
                    }
                });
                
                // 处理按钮
                processBtn.addEventListener('click', processImages);
                
                // 重置按钮
                resetBtn.addEventListener('click', resetAll);
            }
            
            // 处理文件选择
            function handleFileSelect(event, areaNumber) {
                const file = event.target.files[0];
                if (!file) return;
                
                // 检查文件类型
                if (!file.type.startsWith('image/')) {
                    alert('请选择图像文件 (JPG, PNG, GIF)');
                    return;
                }
                
                // 检查文件大小
                if (file.size > maxFileSize) {
                    alert('文件大小不能超过 5MB');
                    return;
                }
                
                // 保存文件引用
                if (areaNumber === 1) {
                    image1 = file;
                } else {
                    image2 = file;
                }
                
                // 显示预览
                showPreview(file, areaNumber);
                
                // 更新处理按钮状态
                updateProcessButton();
            }
            
            // 显示图像预览
            function showPreview(file, areaNumber) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const imgSrc = e.target.result;
                    const preview = areaNumber === 1 ? preview1 : preview2;
                    const uploadArea = areaNumber === 1 ? uploadArea1 : uploadArea2;
                    
                    // 隐藏上传区域文本
                    uploadArea.innerHTML = '';
                    
                    // 创建预览图像
                    const img = document.createElement('img');
                    img.src = imgSrc;
                    img.alt = `预览图像 ${areaNumber}`;
                    
                    preview.innerHTML = '';
                    preview.appendChild(img);
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
            
            // 更新处理按钮状态
            function updateProcessButton() {
                processBtn.disabled = !(image1 && image2);
            }
            
            // 处理图像
            async function processImages() {
                if (!image1 || !image2) {
                    alert('请先上传两张图像');
                    return;
                }
                
                // 显示加载状态
                emptyOutput.style.display = 'none';
                processingResult.style.display = 'none';
                loading.style.display = 'block';
                processBtn.disabled = true;
                
                try {
                    // 创建FormData对象
                    const formData = new FormData();
                    formData.append('image1', image1);
                    formData.append('image2', image2);
                    
                    // 发送请求到后端
                    const response = await fetch('/process-images', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`请求失败: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    
                    // 显示处理结果
                    displayResults(result);
                    
                } catch (error) {
                    console.error('处理图像时出错:', error);
                    alert(`处理图像时出错: ${error.message}`);
                } finally {
                    // 隐藏加载状态
                    loading.style.display = 'none';
                    processBtn.disabled = false;
                }
            }
            
            // 显示处理结果
            function displayResults(result) {
                // 显示结果容器
                emptyOutput.style.display = 'none';
                processingResult.style.display = 'block';
                
                // 显示图像
                resultImages.innerHTML = '';
                
                if (result.image1_preview && result.image2_preview) {
                    const img1Col = document.createElement('div');
                    img1Col.className = 'result-image';
                    img1Col.innerHTML = `
                        <h3>图像 1</h3>
                        <img src="data:image/png;base64,${result.image1_preview}" alt="处理后的图像1">
                    `;
                    
                    const img2Col = document.createElement('div');
                    img2Col.className = 'result-image';
                    img2Col.innerHTML = `
                        <h3>图像 2</h3>
                        <img src="data:image/png;base64,${result.image2_preview}" alt="处理后的图像2">
                    `;
                    
                    resultImages.appendChild(img1Col);
                    resultImages.appendChild(img2Col);
                }
                
                // 显示详细信息
                resultDetails.innerHTML = `
                    <p><strong>分析结果:</strong> ${result.reports}</p>
                `;
            }
            
            // 重置所有
            function resetAll() {
                // 重置图像
                image1 = null;
                image2 = null;
                
                // 重置预览
                preview1.innerHTML = '';
                preview1.style.display = 'none';
                preview2.innerHTML = '';
                preview2.style.display = 'none';
                
                // 重置上传区域文本
                uploadArea1.innerHTML = `
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>点击或拖拽图像到此处上传</p>
                    <p class="file-info">支持 JPG, PNG, GIF 格式，最大 5MB</p>
                `;
                
                uploadArea2.innerHTML = `
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>点击或拖拽图像到此处上传</p>
                    <p class="file-info">支持 JPG, PNG, GIF 格式，最大 5MB</p>
                `;
                
                // 重置输出区域
                emptyOutput.style.display = 'flex';
                processingResult.style.display = 'none';
                loading.style.display = 'none';
                
                // 重置按钮状态
                processBtn.disabled = true;
                
                // 重置文件输入
                fileInput1.value = '';
                fileInput2.value = '';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/process-images")
async def process_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """处理上传的两张图像"""
    try:
        # 检查文件类型
        if not image1.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件1必须是图像")
        if not image2.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件2必须是图像")
        
        # 读取图像数据
        image1_data = await image1.read()
        image2_data = await image2.read()
        
        # 使用PIL打开图像
        image_1 = Image.open(io.BytesIO(image1_data)).convert('RGB')
        image_2 = Image.open(io.BytesIO(image2_data)).convert('RGB')
        
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        image_1 = transform(image_1)
        image_2 = transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        image = image.unsqueeze(0)
        image = image.to(torch.device('cuda:0'))
        output = model(image, mode='sample')
        reports = model.tokenizer.decode_batch(output.cpu().numpy())
        # 返回处理结果（包装为字典，便于前端接收）
        model_t = ChatOpenAI(
        model = "deepseek-chat",
        temperature = 0.3,
        api_key = "sk-c8f09fecf95a49f2b7b1456e7fb5f3e9",
        base_url = "https://api.deepseek.com"
        )
        chat_template = ChatPromptTemplate(
        [
            ("system","你是一个翻译员"),
            ("human","翻译信息如下：{reports}")
        ]
        )
        t_chain = chat_template | model_t
        res = t_chain.invoke(reports).content
        return {"reports": res}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图像时出错: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "image-processing-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)