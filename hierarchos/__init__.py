from .models.core import HierarchosCore
from .models.ltm import LTMModule
from .models.quantized import QuantizedHierarchos, load_quantized
from .training.trainer import train, finetune
from .training.datasets import (
    IterableChunkedJSONLDataset, 
    PTChunkedDataset, 
    StreamingJSONLDataset,
    OriginalJSONLDataset, 
    HuggingFaceStreamingDataset,
    HuggingFaceMapStyleDataset,
    process_text_sample,
    process_tokenized_sample,
    create_dataloader_for_jsonl,
    create_dataloader_for_hf_streaming,
    create_dataloader_for_chunked,
    create_dataloader_pt_chunked,
    create_map_style_dataloader
)
from .utils.device import (
    configure_torch_runtime,
    cuda_diagnostics,
    describe_device,
    pick_device,
    set_threads,
    is_directml_device,
    setup_msvc_environment,
)
from .utils.checkpoint import AttrDict, load_full_model_with_config, save_checkpoint_safely
from .inference.chat import chat

# Optional evaluation support (requires: pip install lm-eval)
try:
    from .evaluation import run_eval, is_lm_eval_available, format_results, HierarchosLM
except ImportError:
    # lm-eval not installed - evaluation features not available
    run_eval = None
    is_lm_eval_available = lambda: False
    format_results = None
    HierarchosLM = None
