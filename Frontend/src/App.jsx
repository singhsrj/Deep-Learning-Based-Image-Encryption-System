import { useState } from 'react';
import { Upload, Lock, Unlock, ArrowRight, Loader2 } from 'lucide-react';

export default function App() {
  const [inputImage, setInputImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);

  // Placeholder images from public folder
  const intermediateImages = {
    substituted: '/substituted_image.png',
    perturbed: '/perturbed.png',
    aesEncrypted: '/aes-encrypted.png',
    aesDecrypted: '/aes-decrypted.png'
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setInputImage(event.target.result);
        setShowResults(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const simulateEncryption = async () => {
    setIsProcessing(true);
    setProgress(0);
    setShowResults(false);

    // Random processing time between 40-50 seconds
    const totalTime = Math.floor(Math.random() * 11000) + 40000;
    setProcessingTime(totalTime);

    const startTime = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const currentProgress = Math.min((elapsed / totalTime) * 100, 100);
      setProgress(currentProgress);

      if (elapsed >= totalTime) {
        clearInterval(interval);
        setIsProcessing(false);
        setShowResults(true);
      }
    }, 100);
  };

  const steps = [
    { name: 'Input Image', image: inputImage, description: 'Original image uploaded by user' },
    { name: 'Substitution', image: intermediateImages.substituted, description: 'Pixel substitution applied' },
    { name: 'Perturbation', image: intermediateImages.perturbed, description: 'Chaotic perturbation applied' },
    { name: 'AES Encryption', image: intermediateImages.aesEncrypted, description: 'AES-256 encryption applied' },
    { name: 'AES Decryption', image: intermediateImages.aesDecrypted, description: 'AES-256 decryption applied' },
    { name: 'Inverse Perturbation', image: intermediateImages.perturbed, description: 'Perturbation reversed' },
    { name: 'Inverse Substitution', image: intermediateImages.substituted, description: 'Substitution reversed' },
    { name: 'Decrypted Image', image: inputImage, description: 'Final decrypted image (matches input)' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Lock className="w-12 h-12 text-purple-400" />
            <h1 className="text-5xl font-bold text-white">Secure Image Encryption</h1>
          </div>
          <p className="text-gray-300 text-lg">
            Advanced encryption pipeline with substitution, perturbation, and AES-256
          </p>
        </div>

        {/* Upload Section */}
        <div className="max-w-2xl mx-auto mb-12">
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20">
            <label className="flex flex-col items-center justify-center cursor-pointer group">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              <div className="w-full border-3 border-dashed border-purple-400 rounded-xl p-12 group-hover:border-purple-300 transition-all group-hover:bg-white/5">
                <Upload className="w-16 h-16 text-purple-400 mx-auto mb-4" />
                <p className="text-white text-xl font-semibold text-center mb-2">
                  Click to upload image
                </p>
                <p className="text-gray-400 text-center">
                  PNG, JPG, GIF up to 10MB
                </p>
              </div>
            </label>

            {inputImage && !isProcessing && !showResults && (
              <div className="mt-6">
                <div className="flex items-center justify-center mb-4">
                  <img
                    src={inputImage}
                    alt="Input"
                    className="max-w-xs max-h-64 rounded-lg shadow-2xl border-2 border-purple-400"
                  />
                </div>
                <button
                  onClick={simulateEncryption}
                  className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold py-4 px-6 rounded-xl transition-all transform hover:scale-105 flex items-center justify-center gap-2"
                >
                  <Lock className="w-5 h-5" />
                  Start Encryption Process
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Processing Loader */}
        {isProcessing && (
          <div className="max-w-2xl mx-auto mb-12">
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 border border-white/20">
              <div className="flex items-center justify-center mb-6">
                <Loader2 className="w-12 h-12 text-purple-400 animate-spin" />
              </div>
              <h3 className="text-2xl font-bold text-white text-center mb-4">
                Processing Encryption...
              </h3>
              <div className="w-full bg-gray-700 rounded-full h-4 mb-4 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-4 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-center text-gray-300">
                {progress.toFixed(1)}% Complete - Applying cryptographic transformations
              </p>
              <p className="text-center text-gray-400 text-sm mt-2">
                Estimated time: {(processingTime / 1000).toFixed(0)} seconds
              </p>
            </div>
          </div>
        )}

        {/* Results Pipeline */}
        {showResults && (
          <div className="space-y-8">
            <h2 className="text-3xl font-bold text-white text-center mb-8">
              Encryption/Decryption Pipeline
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {steps.map((step, index) => (
                <div key={index} className="relative">
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 hover:border-purple-400 transition-all">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-white font-bold">
                        {index + 1}
                      </div>
                      <h3 className="text-white font-semibold text-sm">{step.name}</h3>
                    </div>
                    
                    <div className="mb-4 bg-gray-800 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
                      {step.image ? (
                        <img
                          src={step.image}
                          alt={step.name}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23374151" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%239CA3AF" font-size="16"%3EPlaceholder%3C/text%3E%3C/svg%3E';
                          }}
                        />
                      ) : (
                        <div className="text-gray-500 text-xs">No image</div>
                      )}
                    </div>

                    <p className="text-gray-400 text-xs">{step.description}</p>

                    {index === 3 && (
                      <div className="mt-4 p-2 bg-red-500/20 rounded border border-red-500/50">
                        <p className="text-red-300 text-xs font-semibold flex items-center gap-1">
                          <Lock className="w-3 h-3" />
                          Encrypted State
                        </p>
                      </div>
                    )}

                    {index === 7 && (
                      <div className="mt-4 p-2 bg-green-500/20 rounded border border-green-500/50">
                        <p className="text-green-300 text-xs font-semibold flex items-center gap-1">
                          <Unlock className="w-3 h-3" />
                          Successfully Decrypted
                        </p>
                      </div>
                    )}
                  </div>

                  {index < steps.length - 1 && (
                    <div className="hidden lg:block absolute top-1/2 -right-3 transform -translate-y-1/2 z-10">
                      <ArrowRight className="w-6 h-6 text-purple-400" />
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Key Output Images */}
            <div className="mt-12">
              <h3 className="text-2xl font-bold text-white text-center mb-6">
                Key Transformation Stages
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                  <h4 className="text-purple-400 font-bold mb-3">Substituted</h4>
                  <img
                    src={intermediateImages.substituted}
                    alt="Substituted"
                    className="w-full rounded-lg"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23374151" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%239CA3AF"%3ESubstituted%3C/text%3E%3C/svg%3E';
                    }}
                  />
                </div>

                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                  <h4 className="text-purple-400 font-bold mb-3">Perturbed</h4>
                  <img
                    src={intermediateImages.perturbed}
                    alt="Perturbed"
                    className="w-full rounded-lg"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23374151" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%239CA3AF"%3EPerturbed%3C/text%3E%3C/svg%3E';
                    }}
                  />
                </div>

                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                  <h4 className="text-purple-400 font-bold mb-3">AES Encrypted</h4>
                  <img
                    src={intermediateImages.aesEncrypted}
                    alt="AES Encrypted"
                    className="w-full rounded-lg"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23374151" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%239CA3AF"%3EAES%20Encrypted%3C/text%3E%3C/svg%3E';
                    }}
                  />
                </div>

                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
                  <h4 className="text-purple-400 font-bold mb-3">AES Decrypted</h4>
                  <img
                    src={intermediateImages.aesDecrypted}
                    alt="AES Decrypted"
                    className="w-full rounded-lg"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23374151" width="200" height="200"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%239CA3AF"%3EAES%20Decrypted%3C/text%3E%3C/svg%3E';
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="text-center mt-8">
              <button
                onClick={() => {
                  setShowResults(false);
                  setInputImage(null);
                }}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold py-3 px-8 rounded-xl transition-all transform hover:scale-105"
              >
                Encrypt Another Image
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}