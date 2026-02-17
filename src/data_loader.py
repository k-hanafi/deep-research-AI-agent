"""
Data Loader

Handles loading the Crunchbase dataset, validating companies,
and managing checkpoints for resumable processing.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from pydantic import ValidationError

from .config import DATA_DIR, CHECKPOINT_DIR, PROCESSING
from .models import Company, PipelineCheckpoint, CompanyResult


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and validates the Crunchbase dataset.
    
    Provides iteration over companies with optional filtering
    and batch support for efficient processing.
    """
    
    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to CSV file. Defaults to standard location.
        """
        self.csv_path = csv_path or DATA_DIR / "44k_crunchbase_startups.csv"
        self._df: Optional[pd.DataFrame] = None
        self._validation_errors: list[dict] = []
    
    def load(self) -> "DataLoader":
        """
        Load the CSV into memory.
        
        Returns:
            Self for method chaining.
        
        Raises:
            FileNotFoundError: If CSV doesn't exist.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        
        self._df = pd.read_csv(self.csv_path)
        # Replace NaN with None for cleaner handling
        self._df = self._df.where(pd.notna(self._df), None)
        
        return self
    
    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the raw DataFrame."""
        if self._df is None:
            self.load()
        return self._df
    
    @property
    def total_count(self) -> int:
        """Total number of companies in dataset."""
        return len(self.dataframe)
    
    @property
    def validation_errors(self) -> list[dict]:
        """Companies that failed validation."""
        return self._validation_errors
    
    # ─────────────────────────────────────────────────────────────────────────
    # ITERATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def iter_companies(
        self,
        start_index: int = 0,
        limit: Optional[int] = None
    ) -> Iterator[Company]:
        """
        Iterate over companies as validated Pydantic models.
        
        Args:
            start_index: Skip companies before this index (for resuming).
            limit: Maximum number of companies to yield.
        
        Yields:
            Validated Company objects.
        """
        df = self.dataframe
        
        end_index = len(df) if limit is None else min(start_index + limit, len(df))
        
        for idx in range(start_index, end_index):
            row = df.iloc[idx]
            company = self._row_to_company(row, idx)
            if company:
                yield company
    
    def iter_batches(
        self,
        batch_size: int = PROCESSING.batch_size,
        start_index: int = 0
    ) -> Iterator[list[Company]]:
        """
        Iterate over companies in batches.
        
        Args:
            batch_size: Number of companies per batch.
            start_index: Starting index (for resuming).
        
        Yields:
            Lists of Company objects.
        """
        batch: list[Company] = []
        
        for company in self.iter_companies(start_index=start_index):
            batch.append(company)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining companies
        if batch:
            yield batch
    
    def get_company_by_rcid(self, rcid: int) -> Optional[Company]:
        """
        Retrieve a specific company by its rcid.
        
        Args:
            rcid: The Crunchbase record ID.
        
        Returns:
            Company if found and valid, None otherwise.
        """
        df = self.dataframe
        matches = df[df["rcid"] == rcid]
        
        if matches.empty:
            return None
        
        row = matches.iloc[0]
        idx = matches.index[0]
        return self._row_to_company(row, idx)
    
    def get_companies_by_name(self, name: str, exact: bool = False) -> list[Company]:
        """
        Search for companies by name.
        
        Args:
            name: Company name to search for.
            exact: If True, require exact match. If False, case-insensitive contains.
        
        Returns:
            List of matching companies.
        """
        df = self.dataframe
        
        if exact:
            matches = df[df["name"] == name]
        else:
            matches = df[df["name"].str.contains(name, case=False, na=False)]
        
        companies = []
        for idx, row in matches.iterrows():
            company = self._row_to_company(row, idx)
            if company:
                companies.append(company)
        
        return companies
    
    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def _row_to_company(self, row: pd.Series, idx: int) -> Optional[Company]:
        """
        Convert a DataFrame row to a Company model.
        
        Handles validation errors gracefully by logging them
        and returning None for invalid rows.
        """
        try:
            return Company(
                rcid=int(row["rcid"]),
                org_uuid=str(row["org_uuid"]),
                name=str(row["name"]) if row["name"] else "Unknown",
                cb_url=str(row["cb_url"]),
                homepage_url=row["homepage_url"] if row["homepage_url"] else None,
                short_description=row["short_description"],
                category_list=row["category_list"],
                category_groups_list=row["category_groups_list"],
                created_date=row["created_date"],
                founded_date=row["founded_date"],
                description=row["description"],
            )
        except ValidationError as e:
            self._validation_errors.append({
                "index": idx,
                "rcid": row.get("rcid"),
                "name": row.get("name"),
                "error": str(e)
            })
            return None


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages pipeline checkpoints for resumable processing.
    
    Saves progress periodically so long-running jobs can be
    interrupted and resumed without losing work.
    """
    
    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            run_id: Unique identifier for this run. Auto-generated if not provided.
        """
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{self.run_id}.json"
        self._checkpoint: Optional[PipelineCheckpoint] = None
    
    def create(self, total_companies: int) -> PipelineCheckpoint:
        """
        Create a new checkpoint for a fresh run.
        
        Args:
            total_companies: Total number of companies to process.
        
        Returns:
            New PipelineCheckpoint object.
        """
        now = datetime.utcnow()
        self._checkpoint = PipelineCheckpoint(
            run_id=self.run_id,
            started_at=now,
            last_updated=now,
            total_companies=total_companies,
            processed_count=0,
            last_processed_id=0,
            total_cost_so_far=0.0,
            findings_count=0,
        )
        self._save()
        return self._checkpoint
    
    def load(self) -> Optional[PipelineCheckpoint]:
        """
        Load existing checkpoint from disk.
        
        Returns:
            PipelineCheckpoint if exists, None otherwise.
        """
        if not self.checkpoint_path.exists():
            return None
        
        with open(self.checkpoint_path, "r") as f:
            data = json.load(f)
        
        self._checkpoint = PipelineCheckpoint(**data)
        return self._checkpoint
    
    def load_latest(self) -> Optional[PipelineCheckpoint]:
        """
        Find and load the most recent checkpoint.
        
        Returns:
            Most recent PipelineCheckpoint, or None if none exist.
        """
        checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        
        # Sort by modification time, newest first
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = checkpoints[0]
        
        self.run_id = latest.stem.replace("checkpoint_", "")
        self.checkpoint_path = latest
        
        return self.load()
    
    def update(
        self,
        processed_count: int,
        last_processed_id: int,
        cost_incurred: float = 0.0,
        findings_added: int = 0,
        stage_1_only: int = 0,
        stage_2a_count: int = 0,
        stage_2b_count: int = 0,
        stage_3_count: int = 0,
    ) -> PipelineCheckpoint:
        """
        Update checkpoint with new progress.
        
        Args:
            processed_count: Total companies processed so far.
            last_processed_id: rcid of last processed company.
            cost_incurred: Additional cost since last update.
            findings_added: Number of new findings since last update.
            stage_*: Increment for each stage counter.
        
        Returns:
            Updated checkpoint.
        """
        if self._checkpoint is None:
            raise RuntimeError("No checkpoint initialized. Call create() first.")
        
        # Create updated checkpoint (PipelineCheckpoint is immutable)
        self._checkpoint = PipelineCheckpoint(
            run_id=self._checkpoint.run_id,
            started_at=self._checkpoint.started_at,
            last_updated=datetime.utcnow(),
            total_companies=self._checkpoint.total_companies,
            processed_count=processed_count,
            last_processed_id=last_processed_id,
            total_cost_so_far=self._checkpoint.total_cost_so_far + cost_incurred,
            findings_count=self._checkpoint.findings_count + findings_added,
            stage_1_only=self._checkpoint.stage_1_only + stage_1_only,
            stage_2a_count=self._checkpoint.stage_2a_count + stage_2a_count,
            stage_2b_count=self._checkpoint.stage_2b_count + stage_2b_count,
            stage_3_count=self._checkpoint.stage_3_count + stage_3_count,
        )
        
        self._save()
        return self._checkpoint
    
    def _save(self) -> None:
        """Save current checkpoint to disk."""
        if self._checkpoint is None:
            return
        
        with open(self.checkpoint_path, "w") as f:
            json.dump(self._checkpoint.model_dump(mode="json"), f, indent=2, default=str)
    
    @property
    def checkpoint(self) -> Optional[PipelineCheckpoint]:
        """Current checkpoint state."""
        return self._checkpoint
    
    def should_save(self, processed_count: int) -> bool:
        """
        Check if it's time to save a checkpoint.
        
        Args:
            processed_count: Current number processed.
        
        Returns:
            True if checkpoint should be saved.
        """
        return processed_count % PROCESSING.checkpoint_every == 0


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_companies(limit: Optional[int] = None) -> list[Company]:
    """
    Quick helper to load companies as a list.
    
    Args:
        limit: Maximum number to load (for testing).
    
    Returns:
        List of validated Company objects.
    """
    loader = DataLoader().load()
    return list(loader.iter_companies(limit=limit))


def get_sample_companies(n: int = 10, seed: int = 42) -> list[Company]:
    """
    Get a random sample of companies for testing.
    
    Args:
        n: Number of companies to sample.
        seed: Random seed for reproducibility.
    
    Returns:
        List of sampled Company objects.
    """
    loader = DataLoader().load()
    
    # Sample from dataframe
    sample_df = loader.dataframe.sample(n=n, random_state=seed)
    
    companies = []
    for idx, row in sample_df.iterrows():
        company = loader._row_to_company(row, idx)
        if company:
            companies.append(company)
    
    return companies
