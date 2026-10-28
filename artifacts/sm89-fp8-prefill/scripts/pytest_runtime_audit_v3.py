import hashlib,json,os
from pathlib import Path
import vllm

def pytest_sessionstart(session):
 p=Path(vllm.__file__).parent/'_C_stable_libtorch.abi3.so'
 sha=hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha==os.environ['RESEARCH_CORE_SHA256'],(str(p),sha)
 print(json.dumps(dict(verified_test_runtime=str(p),sha256=sha)),flush=True)
