"""ALGORITHMS data — mathematical definitions for all distillation algorithms."""

ALGORITHMS = [
    {
        "title": "Pipeline Overview",
        "file": "run_distillation_agent.py",
        "description": "Three sequential training stages. Choose Stage 2A/B (forward KL, MLX) or Stage 2C (reverse KL, MiniLLM).",
        "blocks": [
            {
                "label": "End-to-end curriculum",
                "eq": r"\underbrace{\mathcal{L}_{\text{SFT}}}_{\text{Stage 1: warmup}}"
                      r"\;\longrightarrow\;"
                      r"\underbrace{\alpha\,\mathcal{L}_{\text{CE}} + (1-\alpha)\,\mathcal{L}_{\text{KD}}^{\text{fwd}}}_{\text{Stage 2A/B — Forward KL (MLX)}}"
                      r"\;\Big|\;"
                      r"\underbrace{\mathcal{L}_{\text{GRPO}}}_{\text{Stage 2C — Reverse KL (MiniLLM)}}",
            },
            {
                "label": "Forward vs. reverse KL",
                "eq": r"\mathcal{L}_{\text{KD}}^{\text{fwd}} = D_{\text{KL}}(p_T \| p_S)"
                      r"\quad\underbrace{\text{mean-seeking}}_{\text{covers all teacher modes}}"
                      r"\qquad\Big|\qquad"
                      r"\mathcal{L}_{\text{KD}}^{\text{rev}} = D_{\text{KL}}(p_S \| p_T)"
                      r"\quad\underbrace{\text{mode-seeking}}_{\text{sharpens on best mode}}",
            },
        ],
        "params": [],
    },
    {
        "title": "Stage 1 — SFT Warmup",
        "file": "distill_sft.py",
        "description": "Teacher generates greedy completions; student minimises cross-entropy on response tokens only.",
        "blocks": [
            {
                "label": "SFT loss (response tokens only)",
                "eq": r"y^* = \arg\max_y T(y \mid x)"
                      r"\qquad"
                      r"\mathcal{L}_{\text{SFT}} = -\frac{1}{|R|} \sum_{t \in R} \log p_S\!\left(y_t^* \mid y_{<t}^*,\, x\right)",
            },
        ],
        "params": [
            (r"R", "response token positions (prompt + pad masked to $-100$)"),
        ],
    },
    {
        "title": "Stage 2A/B — Forward KL + CE (MLX)",
        "file": "distill_mlx.py",
        "description": (
            "Teacher logits pre-computed once as top-\\(K\\) sparse tensors. "
            "Combined KD + CE loss with optional linear annealing of \\(\\tau\\) and \\(\\alpha\\)."
        ),
        "blocks": [
            {
                "label": "Top-$K$ sparse teacher cache (computed once, frozen)",
                "eq": r"\mathcal{K}(x) = \operatorname{top\text{-}}K\!\left[z_T(x)\right]"
                      r"\;\in\;\mathbb{R}^{B \times T \times K}"
                      r"\qquad"
                      r"\tilde{p}_T^{(\tau)}(k) = \operatorname{softmax}\!\left(\tfrac{z_T^{(K)}(x)}{\tau}\right)",
            },
            {
                "label": "Combined objective",
                "eq": r"\boxed{\mathcal{L} = \alpha_{\text{CE}}\,\mathcal{L}_{\text{CE}} + (1-\alpha_{\text{CE}})\,\mathcal{L}_{\text{KD}}}",
            },
            {
                "label": "Phase 3 — linear annealing",
                "eq": r"\tau(t) = \tau_0 + (\tau_1 - \tau_0)\,\tfrac{t}{T}"
                      r"\qquad"
                      r"\alpha(t) = \alpha_0 + (\alpha_1 - \alpha_0)\,\tfrac{t}{T}",
            },
        ],
        "params": [
            (r"K = 50", r"top-K logits kept ($>$99\% of teacher mass)"),
            (r"\alpha_{\text{CE}} = 0.1", "0 = pure KD, 1 = pure CE"),
            (r"\tau = 1.0", "KD temperature"),
        ],
    },
    {
        "title": "Stage 2C — Reverse KL / GRPO (MiniLLM)",
        "file": "distill_minillm.py",
        "description": (
            "Student policy trained with GRPO: sample \\(G\\) completions, score with reward, "
            "normalise to group advantage, clip importance ratio."
        ),
        "blocks": [
            {
                "label": "Group advantage",
                "eq": r"\{y_i\}_{i=1}^G \sim p_S(\cdot \mid x)"
                      r"\qquad"
                      r"\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \varepsilon}"
                      r"\qquad"
                      r"r(y) = \begin{cases} -1 & |y|<10 \\ +0.5 & \text{otherwise} \end{cases}",
            },
            {
                "label": "GRPO clipped surrogate loss",
                "eq": r"\boxed{\mathcal{L}_{\text{GRPO}} = "
                      r"-\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_t "
                      r"\min\!\Bigl(\rho_{i,t}\hat{A}_i,\;"
                      r"\operatorname{clip}(\rho_{i,t},\,1\!-\!\varepsilon,\,1\!+\!\varepsilon)\hat{A}_i\Bigr)}",
            },
        ],
        "params": [
            (r"G = 4", "completions sampled per prompt"),
            (r"\varepsilon", "PPO clip ratio"),
            (r"\rho_{i,t} = p_S / p_{\text{ref}}", "importance ratio vs frozen reference"),
        ],
    },
    {
        "title": "LoRA Parameterization",
        "file": "all backends",
        "description": (
            "Low-rank adapters on \\(\\{W_Q, W_K, W_V, W_O\\}\\). "
            "\\(W_0\\) is frozen; only \\(A, B\\) are trained."
        ),
        "blocks": [
            {
                "label": "Adapted forward pass",
                "eq": r"h = W_0 x + \frac{\alpha_r}{r}\,B(Ax)"
                      r"\qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times d}"
                      r"\qquad B \leftarrow 0,\; A \sim \mathcal{N}(0,\sigma^2)",
            },
        ],
        "params": [
            (r"r", "rank — 8 (MLX) / 64 (PyTorch)"),
            (r"\alpha_r = 2r", "scale factor = 2.0"),
        ],
    },
    {
        "title": "Optimizer — AdamW + Cosine LR",
        "file": "all backends",
        "description": "AdamW with 3% linear warmup then cosine decay. Gradients clipped to norm 1.0.",
        "blocks": [
            {
                "label": "Parameter update",
                "eq": r"\theta_t = \theta_{t-1} "
                      r"- \eta_t\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} "
                      r"- \eta_t\,\lambda\,\theta_{t-1}",
            },
            {
                "label": "Learning rate schedule",
                "eq": r"\eta_t = \begin{cases} "
                      r"\eta_{\max}\,\dfrac{t}{T_w} & t < T_w \\[6pt] "
                      r"\dfrac{\eta_{\max}}{2}\!\left(1 + \cos\!\pi\,\dfrac{t-T_w}{T-T_w}\right) & t \geq T_w "
                      r"\end{cases}"
                      r"\qquad T_w = \lceil 0.03\,T \rceil",
            },
        ],
        "params": [
            (r"\eta_{\max}", r"2\times10^{-4} (MLX/SFT) \;|\; 2\times10^{-5} (MiniLLM)"),
            (r"\lambda", "weight decay (decoupled)"),
        ],
    },
]
