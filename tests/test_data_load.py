"""Unit tests for data loading modules"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load import load_raw_data, validate_data


class TestDataLoading:
    """Test cases for data loading functionality"""
    
    def test_load_raw_data(self):
        """Test loading raw data from CSV"""
        filepath = "data/raw/sample_data.csv"
        df = load_raw_data(filepath)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "age" in df.columns
        assert "salary" in df.columns
        assert "experience" in df.columns
    
    def test_validate_data_success(self):
        """Test successful data validation"""
        filepath = "data/raw/sample_data.csv"
        df = load_raw_data(filepath)
        
        assert validate_data(df) is True
    
    def test_validate_data_missing_columns(self):
        """Test validation fails with missing columns"""
        df = pd.DataFrame({'name': ['Alice', 'Bob']})
        
        with pytest.raises(ValueError):
            validate_data(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
