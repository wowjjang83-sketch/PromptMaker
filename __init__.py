"""ComfyUI 진입점. ComfyUI/custom_nodes/PromptMaker 폴더에 두면 자동으로 로드됩니다."""
try:
    from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # imported outside ComfyUI (e.g. by test runners)
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
