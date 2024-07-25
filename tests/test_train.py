import pytest
import subprocess

def test_training_script():
  result = subprocess.run(['python','src/train.py'], capture_output = True, text = True)
  assert 'Model trained with MSE' in result.stdout
