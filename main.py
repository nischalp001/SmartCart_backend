from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from queue import Queue
import time
from concurrent.futures import ThreadPoolExecutor

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "best.pt"
MAX_WORKERS = 4            # Adjust based on your GPU memory
BATCH_SIZE = 4             # Optimal batch size for your GPU
INPUT_QUEUE_MAXSIZE = 20   # Prevent memory overload

# -------------------------------
# App setup
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Model load
# -------------------------------
model = YOLO(MODEL_PATH)
model.overrides['batch'] = BATCH_SIZE
model.overrides['verbose'] = False

# -------------------------------
# Queue & processing
# -------------------------------
request_queue = Queue(maxsize=INPUT_QUEUE_MAXSIZE)
result_dict = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def batch_processor():
    """Continuously process requests in batches"""
    while True:
        batch = []
        batch_ids = []

        # Get one item
        item = request_queue.get()
        batch.append(item['image'])
        batch_ids.append(item['id'])

        # Fill up batch
        while len(batch) < BATCH_SIZE and not request_queue.empty():
            try:
                item = request_queue.get_nowait()
                batch.append(item['image'])
                batch_ids.append(item['id'])
            except:
                break

        # Process batch
        try:
            start_time = time.time()
            batch_results = model(batch)

            with lock:
                for img_id, results in zip(batch_ids, batch_results):
                    detections = []
                    for box in results.boxes:
                        detections.append({
                            "class": results.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist()
                        })
                    result_dict[img_id] = detections

            processing_time = time.time() - start_time
            print(f"✅ Processed batch of {len(batch)} in {processing_time:.3f}s "
                  f"({len(batch)/processing_time:.1f} FPS)")

        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            with lock:
                for img_id in batch_ids:
                    result_dict[img_id] = {"error": "Processing failed"}


# Start processing thread
processing_thread = threading.Thread(target=batch_processor, daemon=True)
processing_thread.start()

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
async def home():
    return {
        "status": "🟢 FastAPI inference server is running!",
        "performance": "Optimized for 40+ FPS",
        "config": {
            "max_workers": MAX_WORKERS,
            "batch_size": BATCH_SIZE,
            "queue_size": INPUT_QUEUE_MAXSIZE
        }
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        request_id = str(time.time()) + str(threading.get_ident())

        # Read image into OpenCV format
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Queue check
        if request_queue.qsize() >= INPUT_QUEUE_MAXSIZE:
            return JSONResponse(
                status_code=503,
                content={"error": "Server overloaded - try again later"}
            )

        # Add to processing queue
        request_queue.put({'id': request_id, 'image': img})

        # Wait for result
        start_time = time.time()
        timeout = 5.0
        result = None

        while True:
            with lock:
                if request_id in result_dict:
                    result = result_dict.pop(request_id)
                    break

            if time.time() - start_time > timeout:
                with lock:
                    if request_id in result_dict:
                        result = result_dict.pop(request_id)
                    else:
                        return JSONResponse(
                            status_code=504,
                            content={"error": "Processing timeout"}
                        )
                break

            time.sleep(0.005)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
