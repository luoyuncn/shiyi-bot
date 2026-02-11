"""Tests for built-in tools"""
import pytest
from pathlib import Path
import tempfile
from tools.builtin.file_operations import Tool as FileOpsTool


@pytest.mark.asyncio
async def test_file_read():
    """Test file read operation"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello World")
        temp_path = f.name

    try:
        tool = FileOpsTool()
        result = await tool.run(operation="read", path=temp_path)
        assert "Hello World" in result
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_file_write():
    """Test file write operation"""
    temp_path = tempfile.mktemp(suffix='.txt')

    try:
        tool = FileOpsTool()
        await tool.run(operation="write", path=temp_path, content="Test Content")

        # Verify file was written
        assert Path(temp_path).exists()
        assert Path(temp_path).read_text() == "Test Content"
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
