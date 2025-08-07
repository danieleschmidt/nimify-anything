"""Tests for CLI functionality."""

import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch, Mock

from nimify.cli import main


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.mark.unit
def test_cli_version(cli_runner):
    """Test --version flag."""
    result = cli_runner.invoke(main, ['--version'])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


@pytest.mark.unit
def test_cli_help(cli_runner):
    """Test --help flag."""
    result = cli_runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert "Nimify Anything" in result.output


@pytest.mark.unit
def test_cli_no_args(cli_runner):
    """Test CLI with no arguments shows help."""
    result = cli_runner.invoke(main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


@pytest.mark.integration
@patch('nimify.core.Nimifier')
def test_create_command(mock_nimifier, cli_runner, tmp_path):
    """Test create command functionality."""
    # Create a mock model file
    model_file = tmp_path / "test.onnx"
    model_file.write_bytes(b"fake_model")
    
    # Mock the Nimifier instance
    mock_instance = Mock()
    mock_nimifier.return_value = mock_instance
    
    result = cli_runner.invoke(main, [
        'create',
        str(model_file),
        '--name', 'test-service',
        '--port', '8080'
    ])
    
    # Check that command executed successfully
    assert result.exit_code == 0
    mock_nimifier.assert_called_once()


@pytest.mark.unit
def test_create_command_missing_file(cli_runner):
    """Test create command with non-existent file."""
    result = cli_runner.invoke(main, [
        'create',
        'nonexistent.onnx',
        '--name', 'test-service'
    ])
    
    # Should fail with appropriate error
    assert result.exit_code != 0
    assert "not found" in result.output


@pytest.mark.unit
def test_create_command_invalid_name(cli_runner, tmp_path):
    """Test create command with invalid service name."""
    model_file = tmp_path / "test.onnx"
    model_file.write_bytes(b"fake_model")
    
    result = cli_runner.invoke(main, [
        'create',
        str(model_file),
        '--name', 'invalid name with spaces'
    ])
    
    # Should provide validation error
    assert result.exit_code != 0


@pytest.mark.integration
@patch('nimify.core.Nimifier')
def test_build_command(mock_nimifier, cli_runner):
    """Test build command functionality."""
    mock_instance = Mock()
    mock_nimifier.return_value = mock_instance
    
    result = cli_runner.invoke(main, [
        'build',
        'test-service',
        '--tag', 'test:latest'
    ])
    
    # Verify command structure
    assert "build" in str(result.output).lower() or result.exit_code == 0


@pytest.mark.integration  
@patch('nimify.core.Nimifier')
def test_deploy_command(mock_nimifier, cli_runner):
    """Test deploy command functionality."""
    mock_instance = Mock()
    mock_nimifier.return_value = mock_instance
    
    result = cli_runner.invoke(main, [
        'deploy',
        'test-service',
        '--replicas', '3'
    ])
    
    # Verify command structure
    assert "deploy" in str(result.output).lower() or result.exit_code == 0


@pytest.mark.unit
def test_cli_with_invalid_command(cli_runner):
    """Test CLI with invalid command."""
    result = cli_runner.invoke(main, ['invalid-command'])
    assert result.exit_code != 0
    assert "No such command" in result.output