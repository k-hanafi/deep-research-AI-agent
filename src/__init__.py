"""
AI-Enablement Deep Research Agent

A cost-optimized pipeline for discovering GenAI adoption evidence
across 44k+ startups from Crunchbase.

Architecture:
    Stage 1: Presence Filter (~$0.002/company)
    Stage 2A: Quick Check with Sonar Base (~$0.02)
    Stage 2B: Deep Check with Sonar Pro (~$0.05-0.10)
    Stage 3: Deep Research (selective, ~$0.41-1.19)

Target: < $0.10 average per company
"""

__version__ = "0.1.0"
