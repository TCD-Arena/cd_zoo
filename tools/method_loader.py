"""Centralized method loader for causal discovery methods."""


def load_cd_method(method_name: str):
    """
    Centralized method loader for causal discovery methods.
    
    Args:
        method_name: Name of the method to load
        
    Returns:
        The causal discovery method function
        
    Raises:
        ValueError: If method_name is not in the registry
    """
    METHOD_REGISTRY = {
        "direct_crosscorr": ("methods.baseline_methods", "cross_corr_for_window_causal_graph"),
        "var": ("methods.var", "run_var"),
        "varlingam": ("methods.varlingam", "run_varlingam"),
        "pcmci": ("methods.pcmci", "run_pcmci"),
        "pcmciplus": ("methods.pcmciplus", "run_pcmciplus"),
        "dynotears": ("methods.dynotears", "run_dynotears"),
        "ntsnotears": ("methods.ntsnotears", "run_ntsnotears"),
        "svarrfci": ("methods.svarrfci", "run_svarrfci"),
        "cp": ("methods.causal_pretraining", "run_causal_pretraining"),
        "fpcmci": ("methods.fpcmci", "run_fpcmci"),
        # Additional but not used in the main study
        "physical": ("methods.baseline_methods", "remove_edges_via_mean_values"),
        "combo": ("methods.baseline_methods", "combo_baseline"),
        "crosscorr": ("methods.baseline_methods", "cross_correlation_for_causal_discovery"),
        "cdmi": ("methods.cdmi", "run_cdmi"),
        "svarfci": ("methods.svarfci", "run_svarfci"),

    }
    
    if method_name not in METHOD_REGISTRY:
        available_methods = sorted(METHOD_REGISTRY.keys())
        raise ValueError(
            f"Invalid method: '{method_name}'. "
            f"Available methods: {available_methods}"
        )
    
    module_path, function_name = METHOD_REGISTRY[method_name]
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)
