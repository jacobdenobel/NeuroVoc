
import os
import unittest
import neurovoc
import tempfile
import librosa

import matplotlib.pyplot as plt

class TestGenerate(unittest.TestCase):
    def setUp(self):
        root = os.path.dirname(os.path.dirname(__file__))
        filename = "data/025.wav"
        self.test_file = os.path.join(root, filename)

    def test_specres(self):
        self.assertTrue(os.path.isfile(self.test_file))
        ng = neurovoc.specres(self.test_file, n_trials=1)
        
        print(ng)            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_file.close()  # Windows cannot reopen a file that is still open
        ng.save(temp_file.name)
        
        ng_loaded = neurovoc.Neurogram.load(temp_file.name)
        assert ng.shape == ng_loaded.shape
        print(temp_file.name)
        os.remove(temp_file.name)

    def test_bruce(self):
        self.assertTrue(os.path.isfile(self.test_file))
        ng = neurovoc.bruce(self.test_file, n_trials=1)
        print(ng)      

    def test_reconstruct(self):
        ng = neurovoc.bruce(self.test_file, n_trials=1)
        reconstructed = neurovoc.reconstruct(ng)
        audio_signal, audio_fs = librosa.load(self.test_file, sr=None)

        neurovoc.audio_vs_reconstructed(
            audio_signal,
            reconstructed,
            audio_fs,
            len(ng.frequencies),
            ng.min_freq, 
            ng.max_freq, 
        )
        plt.show()



if __name__ == "__main__":
    unittest.main()
