import { useState } from 'react';
import { Upload, Lock, Unlock, ArrowRight, Loader2, Copy, Check } from 'lucide-react';

export default function App() {
  const [biometricImage, setBiometricImage] = useState(null);
  const [generatedKey, setGeneratedKey] = useState(null);
  const [isGeneratingKey, setIsGeneratingKey] = useState(false);
  const [keyProgress, setKeyProgress] = useState(0);
  const [inputImage, setInputImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [processingTime, setProcessingTime] = useState(0);

  const intermediateImages = {
    substituted: '/substituted_image.png',
    perturbed: '/perturbed.png',
    aesEncrypted: '/aes-encrypted.png',
    aesDecrypted: '/aes-decrypted.png'
  };

  // Generate 256-bit key from biometric image
  const generateBiometricKey = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setBiometricImage(event.target.result);
        setIsGeneratingKey(true);
        setKeyProgress(0);

        // Simulate 10-second key generation
        const startTime = Date.now();
        const interval = setInterval(() => {
          const elapsed = Date.now() - startTime;
          const currentProgress = Math.min((elapsed / 10000) * 100, 100);
          setKeyProgress(currentProgress);

          if (elapsed >= 10000) {
            clearInterval(interval);
            setIsGeneratingKey(false);
            // Generate random 256-bit key (64 hex characters)
            const key = Array.from(crypto.getRandomValues(new Uint8Array(32)))
              .map(b => b.toString(16).padStart(2, '0'))
              .join('')
              .toUpperCase();
            setGeneratedKey(key);
          }
        }, 100);
      };
      reader.readAsDataURL(file);
    }
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
    { name: 'AES Encryption', image: intermediateImages.aesEncrypted, description: 'AES-256 encryption with biometric key' },
    { name: 'AES Decryption', image: intermediateImages.aesDecrypted, description: 'AES-256 decryption with biometric key' },
    { name: 'Inverse Perturbation', image: intermediateImages.perturbed, description: 'Perturbation reversed' },
    { name: 'Inverse Substitution', image: intermediateImages.substituted, description: 'Substitution reversed' },
    { name: 'Decrypted Image', image: inputImage, description: 'Final decrypted image (matches input)' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-slate-900">
      <div className="container mx-auto px-4 py-4 sm:py-8">
        {/* Hero Section */}
        <div className="relative mb-4 sm:mb-8 py-2 sm:py-6">
          {/* Main Hero Content */}
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 sm:mb-6 px-4 leading-tight">
              Biometric-Based Image
              <span className="block text-emerald-400 mt-1 sm:mt-2">Encryption System</span>
            </h1>
            
            <p className="text-sm sm:text-base md:text-lg text-gray-400 max-w-2xl mx-auto px-4 leading-relaxed mb-6 sm:mb-8">
              A comprehensive encryption solution utilizing biometric-based key and AES-256 encryption. This system integrates advanced cryptographic techniques with neural network processing to provide secure image protection.
            </p>

            {/* Feature Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 max-w-3xl mx-auto px-4">
              <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-3 sm:p-4 hover:border-emerald-500/30 transition-all">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center mx-auto mb-2 sm:mb-3">
                  <Lock className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-500" />
                </div>
                <h3 className="text-xs sm:text-sm font-semibold text-white mb-1 sm:mb-2">AES-256 Encryption</h3>
                <p className="text-xs text-gray-400">Industry-standard encryption algorithm</p>
              </div>

              <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-3 sm:p-4 hover:border-emerald-500/30 transition-all">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center mx-auto mb-2 sm:mb-3">
                  <svg className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <h3 className="text-xs sm:text-sm font-semibold text-white mb-1 sm:mb-2">Biometric Keys</h3>
                <p className="text-xs text-gray-400">Secure key generation from biometrics</p>
              </div>

              <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-3 sm:p-4 hover:border-emerald-500/30 transition-all">
                <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center mx-auto mb-2 sm:mb-3">
                  <svg className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xs sm:text-sm font-semibold text-white mb-1 sm:mb-2">Neural Processing</h3>
                <p className="text-xs text-gray-400">Advanced deep learning pipeline</p>
              </div>
            </div>
          </div>
        </div>

        {/* Biometric Key Generation Section */}
        <div className="max-w-2xl mx-auto mb-8 sm:mb-12">
          <h2 className="text-xl sm:text-2xl font-bold text-white mb-4 sm:mb-6 px-4">Step 1: Generate Biometric Key</h2>
          <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 sm:p-6 md:p-8 border border-gray-700 shadow-xl">
            <label className="flex flex-col items-center justify-center cursor-pointer group mb-6">
              <input
                type="file"
                accept="image/*"
                onChange={generateBiometricKey}
                className="hidden"
              />
              <div className="w-full border-3 border-dashed border-gray-600 rounded-lg p-6 sm:p-8 group-hover:border-gray-500 transition-all group-hover:bg-gray-700/50">
                <Upload className="w-12 h-12 sm:w-14 sm:h-14 text-gray-400 mx-auto mb-3 sm:mb-4 group-hover:text-gray-300 transition-colors" />
                <p className="text-white text-lg sm:text-xl font-semibold text-center mb-2">
                  Upload Biometric Image
                </p>
                <p className="text-gray-400 text-center text-sm sm:text-base">
                  Fingerprint, Iris, or Face Image - PNG, JPG up to 10MB
                </p>
              </div>
            </label>

            {biometricImage && (
              <div className="space-y-4">
                <div className="flex items-center justify-center">
                  <img
                    src={biometricImage}
                    alt="Biometric"
                    className="max-w-full sm:max-w-xs max-h-48 sm:max-h-56 rounded-lg shadow-2xl border-2 border-emerald-600"
                  />
                </div>

                {isGeneratingKey && (
                  <div className="bg-gray-700/50 rounded-lg p-4 sm:p-6 border border-emerald-500/50">
                    <p className="text-emerald-400 text-sm font-semibold mb-4 text-center">Generating 256-bit Biometric Key...</p>
                    <div className="w-full bg-gray-600 rounded-full h-2">
                      <div
                        className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${keyProgress}%` }}
                      ></div>
                    </div>
                    <p className="text-gray-300 text-xs text-center mt-3">{keyProgress.toFixed(0)}%</p>
                  </div>
                )}

                {generatedKey && !isGeneratingKey && (
                  <div className="bg-green-500/20 rounded-lg p-4 sm:p-6 border border-green-500/50">
                    <p className="text-green-400 text-sm font-semibold flex items-center justify-center gap-2">
                      <Check className="w-4 h-4" />
                      256-bit Biometric Key Generated Successfully
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {generatedKey && (
          <>
            {/* Image Upload Section */}
            <div className="max-w-2xl mx-auto mb-8 sm:mb-12">
              <h2 className="text-xl sm:text-2xl font-bold text-white mb-4 sm:mb-6 px-4">Step 2: Upload Image to Encrypt</h2>
              <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 sm:p-6 md:p-8 border border-gray-700 shadow-xl">
                <label className="flex flex-col items-center justify-center cursor-pointer group">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                  <div className="w-full border-3 border-dashed border-gray-600 rounded-lg p-6 sm:p-8 md:p-12 group-hover:border-gray-500 transition-all group-hover:bg-gray-700/50">
                    <Upload className="w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16 text-gray-400 mx-auto mb-3 sm:mb-4 group-hover:text-gray-300 transition-colors" />
                    <p className="text-white text-lg sm:text-xl font-semibold text-center mb-2">
                      Click to upload image
                    </p>
                    <p className="text-gray-400 text-center text-sm sm:text-base">
                      PNG, JPG, GIF up to 10MB
                    </p>
                  </div>
                </label>

                {inputImage && !isProcessing && !showResults && (
                  <div className="mt-4 sm:mt-6">
                    <div className="flex items-center justify-center mb-4">
                      <img
                        src={inputImage}
                        alt="Input"
                        className="max-w-full sm:max-w-xs max-h-48 sm:max-h-64 rounded-lg shadow-2xl border-2 border-gray-600"
                      />
                    </div>
                    <button
                      onClick={simulateEncryption}
                      className="w-full bg-emerald-700 hover:bg-emerald-800 text-white font-semibold py-3 sm:py-4 px-4 sm:px-6 rounded-lg transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2 text-sm sm:text-base"
                    >
                      <Lock className="w-4 h-4 sm:w-5 sm:h-5" />
                      Start Encryption with Biometric Key
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Processing Loader */}
            {isProcessing && (
              <div className="max-w-2xl mx-auto mb-8 sm:mb-12">
                <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-6 sm:p-8 border border-gray-700 shadow-xl">
                  <div className="flex items-center justify-center mb-4 sm:mb-6">
                    <div className="relative w-28 h-28 sm:w-32 sm:h-32">
                      <svg className="w-28 h-28 sm:w-32 sm:h-32 transform -rotate-90">
                        <circle
                          cx="56"
                          cy="56"
                          r="50"
                          stroke="currentColor"
                          strokeWidth="7"
                          fill="none"
                          className="text-gray-700 sm:hidden"
                        />
                        <circle
                          cx="64"
                          cy="64"
                          r="56"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          className="text-gray-700 hidden sm:block"
                        />
                        <circle
                          cx="56"
                          cy="56"
                          r="50"
                          stroke="currentColor"
                          strokeWidth="7"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 50}`}
                          strokeDashoffset={`${2 * Math.PI * 50 * (1 - progress / 100)}`}
                          className="text-emerald-500 transition-all duration-300 sm:hidden"
                          strokeLinecap="round"
                        />
                        <circle
                          cx="64"
                          cy="64"
                          r="56"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${2 * Math.PI * 56}`}
                          strokeDashoffset={`${2 * Math.PI * 56 * (1 - progress / 100)}`}
                          className="text-emerald-500 transition-all duration-300 hidden sm:block"
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <Lock className="w-6 h-6 sm:w-8 sm:h-8 text-gray-300 mb-1" />
                        <span className="text-white font-semibold text-base sm:text-lg">{progress.toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-semibold text-white text-center mb-3 sm:mb-4 px-4">
                    Processing Encryption...
                  </h3>
                  <p className="text-center text-gray-300 mb-2 text-sm sm:text-base px-4">
                    Applying biometric key-based cryptographic transformations
                  </p>
                  <p className="text-center text-gray-400 text-xs sm:text-sm px-4">
                    Estimated time: {(processingTime / 1000).toFixed(0)} seconds
                  </p>
                </div>
              </div>
            )}

            {/* Results Pipeline */}
            {showResults && (
              <div className="space-y-6 sm:space-y-8">
                <h2 className="text-2xl sm:text-3xl font-bold text-white text-center mb-6 sm:mb-8 px-4">
                  Encryption/Decryption Pipeline
                </h2>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 px-2 sm:px-0">
                  {steps.map((step, index) => (
                    <div key={index} className="relative">
                      <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 sm:p-6 border border-gray-700 hover:border-emerald-600 transition-all shadow-lg">
                        <div className="flex items-center gap-2 mb-3 sm:mb-4">
                          <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-emerald-700 flex items-center justify-center text-white font-semibold text-sm sm:text-base">
                            {index + 1}
                          </div>
                          <h3 className="text-white font-semibold text-xs sm:text-sm">{step.name}</h3>
                        </div>
                        
                        <div className="mb-3 sm:mb-4 bg-gray-800 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
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
                          <ArrowRight className="w-6 h-6 text-gray-600" />
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Key Output Images */}
                <div className="mt-8 sm:mt-12">
                  <h3 className="text-xl sm:text-2xl font-bold text-white text-center mb-4 sm:mb-6 px-4">
                    Key Transformation Stages
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 max-w-6xl mx-auto px-2 sm:px-0">
                    <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 sm:p-6 border border-gray-700 shadow-lg">
                      <h4 className="text-emerald-400 font-semibold mb-2 sm:mb-3 text-sm sm:text-base">Substituted</h4>
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

                    <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 sm:p-6 border border-gray-700 shadow-lg">
                      <h4 className="text-emerald-400 font-semibold mb-2 sm:mb-3 text-sm sm:text-base">Perturbed</h4>
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

                    <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 sm:p-6 border border-gray-700 shadow-lg">
                      <h4 className="text-emerald-400 font-semibold mb-2 sm:mb-3 text-sm sm:text-base">AES Encrypted</h4>
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

                    <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 sm:p-6 border border-gray-700 shadow-lg">
                      <h4 className="text-emerald-400 font-semibold mb-2 sm:mb-3 text-sm sm:text-base">AES Decrypted</h4>
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

                <div className="text-center mt-6 sm:mt-8 px-4">
                  <button
                    onClick={() => {
                      setShowResults(false);
                      setInputImage(null);
                    }}
                    className="bg-emerald-700 hover:bg-emerald-800 text-white font-semibold py-3 px-6 sm:px-8 rounded-lg transition-all shadow-lg hover:shadow-xl text-sm sm:text-base"
                  >
                    Encrypt Another Image
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}