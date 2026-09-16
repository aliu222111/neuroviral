"""NeuroViral — fully-local, research-informed social-media virality predictor.

This package extracts real audio/visual signals from a short-form video and maps
them onto a research-grounded *simulation* of five neural/attention channels
(NAcc reward, salience/attention, amygdala-insula arousal, anterior-insula
aversion, MPFC social-meaning). Channel dynamics translate into TikTok / Reels
virality scores plus timestamped, specific recommendations.

IMPORTANT HONESTY NOTE: this is NOT a real brain scan. It is a research-informed
proxy grounded in neuroforecasting literature (Falk et al. 2012; Tong et al.
2020 PNAS). See README.md.
"""

__version__ = "0.1.0"

CHANNELS = ("reward", "attention", "emotion", "aversion", "language")

__all__ = ["__version__", "CHANNELS"]
