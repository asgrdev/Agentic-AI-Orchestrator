# راهنمای جامع UI پیشرفته
# Advanced UI Visualization Guide

## 📋 فهرست مطالب

1. [معرفی](#معرفی)
2. [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
3. [تب نظارت بر سنسورها](#تب-نظارت-بر-سنسورها)
4. [تب گراف دانش](#تب-گراف-دانش)
5. [تب فلوی Agent](#تب-فلوی-agent)
6. [API Endpoints](#api-endpoints)
7. [WebSocket](#websocket)
8. [عیب‌یابی](#عیب‌یابی)

---

## معرفی

UI پیشرفته Agentic Graph RAG یک رابط کاربری تحت وب برای نظارت و مدیریت سیستم است که شامل:

✅ **نظارت بر سنسورها**: مشاهده real-time عملکرد سنسورهای Vision, Audio, Text

✅ **گراف دانش**: نمایش بصری گراف دانش و روابط معنایی

✅ **فلوی Agent**: نمایش فلوی اجرای agent به سبک LangGraph

✅ **بروزرسانی Real-time**: استفاده از WebSocket برای بروزرسانی لحظه‌ای

---

## نصب و راه‌اندازی

### 1. نصب Dependencies

```bash
# نصب FastAPI و Uvicorn
pip install fastapi uvicorn[standard] websockets

# یا از requirements.txt
pip install -r requirements.txt
```

### 2. راه‌اندازی Backend API

```bash
# از دایرکتوری اصلی پروژه
cd api
python visualization_api.py
```

یا با uvicorn:

```bash
uvicorn api.visualization_api:app --reload --host 0.0.0.0 --port 8000
```

### 3. باز کردن UI

```
http://localhost:8000/static/index.html
```

یا اگر از FastAPI static files استفاده می‌کنید:

```python
# در visualization_api.py اضافه کنید:
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="api/static"), name="static")
```

سپس:
```
http://localhost:8000/static/index.html
```

---

## تب نظارت بر سنسورها

### ویژگی‌ها

این تب اطلاعات زیر را نمایش می‌دهد:

#### 1. Vision Sensor (👁️)
- **وضعیت**: active, idle, processing, error
- **مدل**: YOLO (object detection)
- **اطمینان**: درصد confidence تشخیص
- **زمان پردازش**: مدت زمان پردازش تصویر
- **آخرین ورودی**: مسیر فایل تصویر

#### 2. Audio Sensor (🎤)
- **وضعیت**: active, idle, processing, error
- **مدل**: Whisper (speech-to-text)
- **اطمینان**: درصد confidence تشخیص
- **زمان پردازش**: مدت زمان پردازش صدا
- **آخرین ورودی**: مسیر فایل صوتی

#### 3. Text Sensor (📝)
- **وضعیت**: active, idle, processing, error
- **مدل**: BERT (embeddings)
- **اطمینان**: درصد confidence
- **زمان پردازش**: مدت زمان پردازش متن
- **آخرین ورودی**: متن ورودی

### نحوه استفاده

```javascript
// پردازش ورودی سنسور
async function processSensorInput(sensorType, inputData) {
    const response = await fetch(`${API_BASE}/sensors/${sensorType}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: inputData })
    });
    
    return await response.json();
}

// مثال
await processSensorInput('vision', { input: 'path/to/image.jpg' });
```

### رنگ‌بندی وضعیت

- 🟢 **Active** (سبز): سنسور فعال و آماده
- ⚪ **Idle** (خاکستری): سنسور در حالت استراحت
- 🟡 **Processing** (زرد): در حال پردازش
- 🔴 **Error** (قرمز): خطا در سنسور

---

## تب گراف دانش

### ویژگی‌ها

#### آمار گراف

- **تعداد Entities**: تعداد کل موجودیت‌ها
- **تعداد Relations**: تعداد کل روابط
- **روابط صریح**: روابط استخراج شده از متن
- **روابط استنتاجی**: روابط تولید شده توسط inference engine
- **کیفیت گراف**: امتیاز کیفیت (0-100%)
- **تراکم**: نسبت روابط به entities

#### نمایش بصری

گراف به صورت interactive نمایش داده می‌شود:

- **Nodes**: دایره‌های آبی (entities)
- **Edges**: خطوط خاکستری (relations)
- **Labels**: نام entities

### کنترل‌ها

```javascript
// بروزرسانی گراف
refreshGraph()

// بزرگنمایی
zoomIn()

// کوچک‌نمایی
zoomOut()

// بازنشانی نما
resetView()
```

### API Usage

```javascript
// دریافت آمار
const stats = await fetch(`${API_BASE}/graph/stats`).then(r => r.json());

// دریافت nodes
const nodes = await fetch(`${API_BASE}/graph/nodes?limit=100`).then(r => r.json());

// دریافت edges
const edges = await fetch(`${API_BASE}/graph/edges?limit=100`).then(r => r.json());

// جستجو در گراف
const results = await fetch(`${API_BASE}/graph/search?query=Python`).then(r => r.json());

// دریافت subgraph
const subgraph = await fetch(`${API_BASE}/graph/subgraph?entity_id=entity_123&depth=2`).then(r => r.json());
```

### مثال: نمایش گراف با D3.js

```html
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
async function drawGraphWithD3() {
    const nodes = await fetch(`${API_BASE}/graph/nodes`).then(r => r.json());
    const edges = await fetch(`${API_BASE}/graph/edges`).then(r => r.json());
    
    const svg = d3.select("#graph-viz");
    const width = 800;
    const height = 600;
    
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(edges).id(d => d.id))
        .force("charge", d3.forceManyBody().strength(-100))
        .force("center", d3.forceCenter(width / 2, height / 2));
    
    // Draw edges
    const link = svg.append("g")
        .selectAll("line")
        .data(edges)
        .enter().append("line")
        .attr("stroke", "#999")
        .attr("stroke-width", 2);
    
    // Draw nodes
    const node = svg.append("g")
        .selectAll("circle")
        .data(nodes)
        .enter().append("circle")
        .attr("r", 10)
        .attr("fill", "#667eea");
    
    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
        
        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
    });
}
</script>
```

---

## تب فلوی Agent

### ویژگی‌ها

این تب فلوی اجرای agent را به سبک LangGraph نمایش می‌دهد:

#### اطلاعات فلو

- **Query**: سوال کاربر
- **تعداد مراحل**: تعداد steps اجرا شده
- **مدت زمان**: زمان کل اجرا

#### Nodes

هر node نمایانگر یک مرحله از فلو است:

- **Start** (🟢): شروع فلو
- **Agent** (🔵): اجرای یک agent
- **Tool** (🟡): فراخوانی یک tool
- **Decision** (🟣): تصمیم‌گیری
- **End** (🔴): پایان فلو

#### وضعیت Nodes

- **Pending** (خاکستری): در انتظار اجرا
- **Active** (آبی متحرک): در حال اجرا
- **Completed** (سبز): اجرا شده
- **Error** (قرمز): خطا

### کنترل‌ها

```javascript
// شروع فلوی جدید
startNewFlow()

// توقف فلو
pauseFlow()

// ادامه فلو
resumeFlow()

// بازنشانی فلو
resetFlow()
```

### API Usage

```javascript
// شروع فلوی جدید
const result = await fetch(`${API_BASE}/flow/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'What is Python?' })
}).then(r => r.json());

// دریافت فلوی فعلی
const flow = await fetch(`${API_BASE}/flow/current`).then(r => r.json());

// بروزرسانی یک step
await fetch(`${API_BASE}/flow/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        node_id: 'understand',
        status: 'completed',
        metadata: { result: 'Plan created' }
    })
});

// دریافت تاریخچه
const history = await fetch(`${API_BASE}/flow/history?limit=10`).then(r => r.json());
```

### مثال: ایجاد فلوی کامل

```javascript
async function createCompleteFlow(query) {
    // 1. شروع فلو
    await fetch(`${API_BASE}/flow/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    });
    
    // 2. مراحل مختلف
    const steps = [
        { id: 'understand', label: 'Understand & Plan', duration: 1000 },
        { id: 'retrieve', label: 'Retrieve Context', duration: 1500 },
        { id: 'reason', label: 'Reason', duration: 2000 },
        { id: 'generate', label: 'Generate Answer', duration: 1000 },
    ];
    
    for (const step of steps) {
        // شروع step
        await fetch(`${API_BASE}/flow/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: step.id,
                status: 'active',
                metadata: {}
            })
        });
        
        // شبیه‌سازی پردازش
        await new Promise(resolve => setTimeout(resolve, step.duration));
        
        // اتمام step
        await fetch(`${API_BASE}/flow/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: step.id,
                status: 'completed',
                metadata: { result: `${step.label} completed` }
            })
        });
    }
}
```

---

## API Endpoints

### Sensor Endpoints

| Method | Endpoint | توضیحات |
|--------|----------|---------|
| GET | `/api/sensors/status` | دریافت وضعیت تمام سنسورها |
| GET | `/api/sensors/{type}` | دریافت وضعیت یک سنسور |
| POST | `/api/sensors/{type}/process` | پردازش ورودی سنسور |

### Graph Endpoints

| Method | Endpoint | توضیحات |
|--------|----------|---------|
| GET | `/api/graph/stats` | آمار گراف دانش |
| GET | `/api/graph/nodes` | لیست nodes |
| GET | `/api/graph/edges` | لیست edges |
| GET | `/api/graph/subgraph` | subgraph اطراف entity |
| GET | `/api/graph/search` | جستجو در گراف |

### Flow Endpoints

| Method | Endpoint | توضیحات |
|--------|----------|---------|
| GET | `/api/flow/current` | فلوی فعلی |
| GET | `/api/flow/history` | تاریخچه فلوها |
| POST | `/api/flow/start` | شروع فلوی جدید |
| POST | `/api/flow/step` | بروزرسانی step |

### Utility Endpoints

| Method | Endpoint | توضیحات |
|--------|----------|---------|
| GET | `/api/health` | بررسی سلامت API |
| POST | `/api/reset` | ریست state |

---

## WebSocket

### اتصال

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    console.log('Connected');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'sensor_update') {
        updateSensorCard(data.sensor, data.data);
    } else if (data.type === 'flow_update') {
        updateFlowVisualization(data.data);
    }
};

ws.onclose = () => {
    console.log('Disconnected');
    // Reconnect
    setTimeout(initWebSocket, 5000);
};
```

### پیام‌های WebSocket

#### Sensor Update
```json
{
    "type": "sensor_update",
    "sensor": "vision",
    "data": {
        "sensor_type": "vision",
        "status": "processing",
        "confidence": 0.95,
        "processing_time": 0.5
    }
}
```

#### Flow Update
```json
{
    "type": "flow_update",
    "data": {
        "nodes": [...],
        "edges": [...],
        "current_node": "retrieve",
        "step_count": 2
    }
}
```

### Heartbeat

```javascript
// ارسال ping هر 30 ثانیه
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
    }
}, 30000);
```

---

## عیب‌یابی

### مشکل: UI باز نمی‌شود

**راه‌حل:**
```bash
# بررسی اجرای API
curl http://localhost:8000/api/health

# بررسی static files
ls api/static/index.html
```

### مشکل: WebSocket متصل نمی‌شود

**راه‌حل:**
```javascript
// بررسی URL
console.log('WS URL:', WS_URL);

// بررسی firewall
# در macOS
sudo pfctl -d

# در Linux
sudo ufw allow 8000
```

### مشکل: داده‌ها نمایش داده نمی‌شوند

**راه‌حل:**
```javascript
// بررسی console
console.log('API Base:', API_BASE);

// تست endpoint
fetch(`${API_BASE}/sensors/status`)
    .then(r => r.json())
    .then(data => console.log(data))
    .catch(err => console.error(err));
```

### مشکل: CORS Error

**راه‌حل:**
```python
# در visualization_api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # آدرس frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## مثال‌های کاربردی

### مثال 1: نظارت مستمر بر سنسورها

```javascript
async function monitorSensors() {
    setInterval(async () => {
        const sensors = await fetch(`${API_BASE}/sensors/status`)
            .then(r => r.json());
        
        sensors.forEach(sensor => {
            if (sensor.status === 'error') {
                alert(`خطا در سنسور ${sensor.sensor_type}`);
            }
        });
    }, 5000);
}
```

### مثال 2: نمایش گراف با فیلتر

```javascript
async function showFilteredGraph(entityType) {
    const nodes = await fetch(`${API_BASE}/graph/nodes`)
        .then(r => r.json());
    
    const filtered = nodes.filter(n => n.type === entityType);
    
    drawGraph(filtered);
}
```

### مثال 3: ذخیره تاریخچه فلو

```javascript
async function saveFlowHistory() {
    const history = await fetch(`${API_BASE}/flow/history?limit=100`)
        .then(r => r.json());
    
    const json = JSON.stringify(history, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flow_history.json';
    a.click();
}
```

---

## نتیجه‌گیری

UI پیشرفته Agentic Graph RAG یک ابزار قدرتمند برای:

✅ نظارت real-time بر سنسورها
✅ نمایش بصری گراف دانش
✅ ردیابی فلوی agent
✅ دیباگ و عیب‌یابی
✅ تحلیل عملکرد سیستم

### نکات مهم

- از WebSocket برای بروزرسانی real-time استفاده کنید
- API را در production با authentication محافظت کنید
- از caching برای بهبود performance استفاده کنید
- logs را برای debugging نگه دارید

---

 