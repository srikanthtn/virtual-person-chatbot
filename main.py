"""
Extended Virtual Memory Base Structure
- FastAPI backend
- Endpoints to upload memories (video/audio/photo/text)
- Local file storage (./storage)
- SQLite metadata DB via SQLModel
- Vector store using Chroma (with fallback to in-memory)
- Real embeddings using sentence-transformers (all-MiniLM-L6-v2)
- Audio transcription using OpenAI Whisper (openai-whisper)
- Video keyframe extraction using OpenCV with scene-change heuristic

Install required packages (suggested):
pip install fastapi "uvicorn[standard]" sqlmodel python-multipart aiofiles pydantic[dotenv] sentence-transformers chromadb openai-whisper opencv-python ffmpeg-python

Notes:
- Whisper can be slow/large; consider using faster-whisper or a cloud ASR for production.
- Chroma will persist vectors under CHROMA_DB_DIR if configured.
- This is a starting point; tune models, concurrency, and error handling for production.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import SQLModel, Field, Session, create_engine, select
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import os
import shutil
from datetime import datetime
import asyncio
from transformers import pipeline
import torch

# ML imports
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import whisper
import cv2
import numpy as np

# -----------------------------
# Config / storage paths
# -----------------------------
STORAGE_DIR = "./storage"
MEDIA_DIR = os.path.join(STORAGE_DIR, "media")
DB_FILE = "memories.db"
CHROMA_DB_DIR = "./chroma_db"
os.makedirs(MEDIA_DIR, exist_ok=True)

# -----------------------------
# Database models (SQLModel)
# -----------------------------
class Memory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str
    created_at: datetime
    kind: str  # 'video'|'audio'|'image'|'text'
    filename: Optional[str] = None
    transcript: Optional[str] = None
    caption: Optional[str] = None
    text: Optional[str] = None
    embedding_id: Optional[str] = None

# create engine
engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)

# -----------------------------
# Initialize models (sentence-transformers + whisper + chroma client)
# -----------------------------
print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Loaded embedding model.")

print("Loading Whisper small model (this may take time)...")
whisper_model = whisper.load_model("small")
print("Loaded Whisper model.")

# chroma client (persistence)
try:
    client = chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))
    collection = client.get_or_create_collection("memories")
    chroma_ready = True
    print("Chroma client ready, persistence at:", CHROMA_DB_DIR)
except Exception as e:
    print("Chroma init failed, falling back to in-memory store:", e)
    client = None
    collection = None
    chroma_ready = False

# -----------------------------
# Utility: embed text
# -----------------------------
async def embed_text(text: str) -> List[float]:
    # sentence-transformers returns numpy array
    vec = st_model.encode(text, show_progress_bar=False)
    return vec.tolist()

# -----------------------------
# Utility: transcribe audio/video using whisper
# -----------------------------
async def transcribe_audio(file_path: str) -> str:
    # whisper is synchronous; run in threadpool
    loop = asyncio.get_event_loop()
    def run_whisper():
        result = whisper_model.transcribe(file_path)
        return result.get("text", "")
    text = await loop.run_in_executor(None, run_whisper)
    return text

# -----------------------------
# Utility: extract keyframes from video (scene-change heuristic)
# -----------------------------
async def extract_keyframes(video_path: str, max_frames: int = 8) -> List[str]:
    """Detect scene changes by histogram difference and save keyframes.
    Returns list of saved keyframe file paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    hist_prev = None
    saved = []
    idx = 0
    step = max(1, total_frames // (max_frames * 4 + 1))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            # compute histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if hist_prev is None:
                hist_prev = hist
                frames.append((idx, frame))
            else:
                diff = cv2.compareHist(hist_prev, hist, cv2.HISTCMP_BHATTACHARYYA)
                # larger diff means scene change; threshold tuned loosely
                if diff > 0.3 and len(frames) < max_frames:
                    frames.append((idx, frame))
                    hist_prev = hist
        idx += 1

    cap.release()

    # if heuristic didn't find enough frames, sample uniformly
    if len(frames) == 0 and total_frames > 0:
        cap = cv2.VideoCapture(video_path)
        sample_idxs = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
        saved = []
        for s in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
            ret, frame = cap.read()
            if not ret: continue
            out_path = os.path.join(MEDIA_DIR, f"{uuid.uuid4()}_keyframe.jpg")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
        cap.release()
        return saved

    # save frames to disk
    for idx, frame in frames[:max_frames]:
        out_path = os.path.join(MEDIA_DIR, f"{uuid.uuid4()}_keyframe.jpg")
        cv2.imwrite(out_path, frame)
        saved.append(out_path)

    return saved

# -----------------------------
# Chroma helpers
# -----------------------------
def chroma_add(emb_id: str, vector: List[float], metadata: Dict):
    if not chroma_ready or collection is None:
        return False
    collection.add(ids=[emb_id], embeddings=[vector], metadatas=[metadata], documents=[metadata.get("text","")])
    client.persist()
    return True

def chroma_query(vector: List[float], top_k: int = 5):
    if not chroma_ready or collection is None:
        return []
    res = collection.query(query_embeddings=[vector], n_results=top_k, include=['metadatas','distances','ids','documents'])
    # res is dict-like
    out = []
    if res and len(res.get('ids', []))>0:
        ids = res['ids'][0]
        metadatas = res['metadatas'][0]
        distances = res['distances'][0]
        docs = res.get('documents',[[]])[0]
        for i, sid in enumerate(ids):
            out.append({"id": sid, "metadata": metadatas[i], "distance": distances[i], "document": docs[i] if i < len(docs) else None})
    return out

# Fallback in-memory store if chroma unavailable
in_memory_store: Dict[str, Dict] = {}

def inmem_add(emb_id: str, vector: List[float], metadata: Dict):
    in_memory_store[emb_id] = {"vector": vector, "metadata": metadata}

def inmem_query(vector: List[float], top_k: int = 5):
    def cosine(a,b):
        a = np.array(a); b = np.array(b)
        if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
            return 0
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    scores = []
    for k,v in in_memory_store.items():
        scores.append((k, cosine(vector, v['vector']), v['metadata']))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [{"id": s[0], "metadata": s[2], "score": s[1]} for s in scores[:top_k]]

# -----------------------------
# FastAPI app and endpoints
# -----------------------------
app = FastAPI(title="Virtual Memory - Extended Service")

@app.on_event("startup")
async def startup_event():
    init_db()

class UploadResponse(BaseModel):
    memory_id: int
    uuid: str






@app.post("/memories/upload", response_model=UploadResponse)
async def upload_memory(kind: str = Form(...), file: UploadFile = File(None), text: str = Form(None)):
    if kind not in {"video","audio","image","text"}:
        raise HTTPException(status_code=400, detail="invalid kind")

    mem_uuid = str(uuid.uuid4())
    filename_on_disk = None
    transcript = None
    caption = None

    # if text memory
    if kind == "text":
        if not text:
            raise HTTPException(status_code=400, detail="text required for kind=text")
        transcript = text

    else:
        if not file:
            raise HTTPException(status_code=400, detail="file required for media kinds")
        # save to disk
        ext = os.path.splitext(file.filename)[1]
        filename_on_disk = f"{mem_uuid}{ext}"
        target_path = os.path.join(MEDIA_DIR, filename_on_disk)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # process depending on kind
        if kind == "audio":
            transcript = await transcribe_audio(target_path)
        elif kind == "image":
            # simple caption: embed filename as placeholder; in prod use BLIP/CLIP captioning
            caption = f"Image saved as {filename_on_disk}"
        elif kind == "video":
            # extract audio (if present) and transcribe; extract keyframes for captions
            transcript = await transcribe_audio(target_path)
            keyframes = await extract_keyframes(target_path, max_frames=6)
            caption = f"Extracted {len(keyframes)} keyframes"

    # Create DB entry
    mem = Memory(
        uuid=mem_uuid,
        created_at=datetime.utcnow(),
        kind=kind,
        filename=filename_on_disk,
        transcript=transcript,
        caption=caption,
        text=text if kind=="text" else None,
    )
    with Session(engine) as session:
        session.add(mem)
        session.commit()
        session.refresh(mem)

    # create embedding from available text/caption/transcript
    content_to_embed = transcript or caption or text or ""
    if content_to_embed:
        emb = await embed_text(content_to_embed)
        emb_id = f"emb_{mem.uuid}"
        metadata = {"memory_id": mem.id, "kind": kind, "created_at": mem.created_at.isoformat()}
        if chroma_ready:
            chroma_add(emb_id, emb, metadata)
        else:
            inmem_add(emb_id, emb, metadata)
        # save embedding id to DB
        with Session(engine) as session:
            db_mem = session.get(Memory, mem.id)
            db_mem.embedding_id = emb_id
            session.add(db_mem)
            session.commit()

    return UploadResponse(memory_id=mem.id, uuid=mem.uuid)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5



# load a free instruct model (small enough for demo, can swap with Mistral/LLaMA)
try:
    print("Loading text generation model...")
    generator = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",  # or "mistralai/Mistral-7B-Instruct-v0.2"
        device_map="auto",  # will use GPU if available
        torch_dtype=torch.float16
    )
    print("Text generation model loaded successfully.")
except Exception as e:
    print(f"Failed to load Falcon model: {e}")
    print("Falling back to a smaller model...")
    try:
        # Fallback to a smaller, more compatible model
        generator = pipeline(
            "text-generation",
            model="gpt2",  # Much smaller and more compatible
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Fallback model (GPT-2) loaded successfully.")
    except Exception as e2:
        print(f"Failed to load fallback model: {e2}")
        print("Text generation will be disabled.")
        generator = None



@app.post("/memories/query")
async def query_memories(q: QueryRequest):
    # embed query
    qvec = await embed_text(q.query)
    if chroma_ready:
        results = chroma_query(qvec, top_k=q.top_k)
        results = [(r["id"], r["metadata"]) for r in results]
    else:
        results = inmem_query(qvec, top_k=q.top_k)
        results = [(r["id"], r["metadata"]) for r in results]

    out = []
    context_texts = []
    with Session(engine) as session:
        for emb_id, meta in results:
            mem = session.get(Memory, meta["memory_id"]) if meta else None
            if mem:
                content = mem.transcript or mem.caption or mem.text or ""
                context_texts.append(content)
                out.append({
                    "memory_id": mem.id,
                    "uuid": mem.uuid,
                    "kind": mem.kind,
                    "created_at": mem.created_at.isoformat(),
                    "transcript": mem.transcript,
                    "caption": mem.caption,
                    "filename": mem.filename,
                })

    # Build context for LLM
    context = "\n".join(context_texts)
    prompt = f"You are a helpful assistant with access to personal memories.\n" \
             f"User query: {q.query}\n" \
             f"Relevant memories: {context}\n" \
             f"Answer the query using the memories above."

    # Generate answer
    if generator is not None:
        try:
            llm_response = generator(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
        except Exception as e:
            print(f"Error generating response: {e}")
            llm_response = "Text generation is currently unavailable. Here are the relevant memories found."
    else:
        llm_response = "Text generation is currently unavailable. Here are the relevant memories found."

    return JSONResponse(content={
        "results": out,
        "llm_answer": llm_response
    })
# -----------------------------
# Simple health / list endpoints
# -----------------------------
@app.get("/memories/list")
async def list_memories():
    with Session(engine) as session:
        res = session.exec(select(Memory)).all()
    return res

@app.get("/health")
async def health():
    return {"status": "ok", "chroma_ready": chroma_ready}

# -----------------------------
# To run locally:
# uvicorn virtual_memory_base_structure:app --reload --port 8000
# -----------------------------
