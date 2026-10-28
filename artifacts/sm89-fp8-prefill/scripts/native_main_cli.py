"""Launch only after proving which pinned-main Python/native pair was imported."""
import hashlib
import json
import os
from pathlib import Path
import vllm
import vllm._C_stable_libtorch as core

package=Path(vllm.__file__).resolve().parent
assert package==Path(os.environ['RESEARCH_RUNTIME_ROOT']).resolve()/'vllm',package
sha=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
assert sha==os.environ['RESEARCH_CORE_SHA256'],(core.__file__,sha)
print(json.dumps({'verified_runtime_package':str(package),'core_library':core.__file__,
                  'core_sha256':sha,'python_version_string':vllm.__version__}),flush=True)

if __name__=='__main__':
    from vllm.entrypoints.cli.main import main
    main()
