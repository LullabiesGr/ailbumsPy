import { useState } from 'react'
import * as Checkbox from '@radix-ui/react-checkbox'
import * as Progress from '@radix-ui/react-progress'
import * as Select from '@radix-ui/react-select'
import * as Slider from '@radix-ui/react-slider'
import axios from 'axios'

function App() {
  const [images, setImages] = useState([])
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState([])
  const [error, setError] = useState(null)
  const [filters, setFilters] = useState({
    eyes: true,
    smile: true,
    duplicates: true,
    minScore: 5
  })

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files)
    setImages(files)
    setError(null) // Clear any previous errors
  }

  const startProcessing = async () => {
    setProcessing(true)
    setError(null)
    const processedResults = []
    
    for (let i = 0; i < images.length; i++) {
      const formData = new FormData()
      formData.append('file', images[i])
      
      try {
        const response = await axios.post('http://localhost:8000/cull', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        processedResults.push({
          file: images[i],
          analysis: response.data
        })
        
        setProgress(((i + 1) / images.length) * 100)
      } catch (error) {
        console.error('Error processing image:', {
          message: error.message,
          status: error.response?.status,
          data: error.response?.data
        })
        setError(`Failed to process image: ${error.message}. Please ensure the backend server is running.`)
        setProcessing(false)
        return
      }
    }
    
    setResults(processedResults)
    setProcessing(false)
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <header className="bg-white rounded-lg shadow-sm p-6 mb-8">
        <h1 className="text-2xl font-bold text-blue-600">Ailbums Web</h1>
        <p className="text-gray-600">Photo culling made simple</p>
      </header>

      <main className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Import Photos</h2>
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleFileSelect}
            className="mb-4"
          />
          
          {error && (
            <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-4">
              {error}
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <h3 className="font-medium mb-2">Filters</h3>
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <Checkbox.Root
                    checked={filters.eyes}
                    onCheckedChange={(checked) => 
                      setFilters(f => ({...f, eyes: checked}))
                    }
                    className="w-5 h-5 border rounded"
                  />
                  Filter closed eyes
                </label>
                <label className="flex items-center gap-2">
                  <Checkbox.Root
                    checked={filters.smile}
                    onCheckedChange={(checked) => 
                      setFilters(f => ({...f, smile: checked}))
                    }
                    className="w-5 h-5 border rounded"
                  />
                  Filter no smile
                </label>
                <label className="flex items-center gap-2">
                  <Checkbox.Root
                    checked={filters.duplicates}
                    onCheckedChange={(checked) => 
                      setFilters(f => ({...f, duplicates: checked}))
                    }
                    className="w-5 h-5 border rounded"
                  />
                  Filter duplicates
                </label>
              </div>
            </div>

            <div>
              <h3 className="font-medium mb-2">Quality Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm mb-1">Minimum Score</label>
                  <Slider.Root
                    value={[filters.minScore]}
                    onValueChange={([value]) => 
                      setFilters(f => ({...f, minScore: value}))
                    }
                    max={10}
                    step={1}
                    className="relative flex items-center w-48 h-5"
                  >
                    <Slider.Track className="bg-gray-200 relative grow h-1 rounded-full">
                      <Slider.Range className="absolute bg-blue-500 h-full rounded-full" />
                    </Slider.Track>
                    <Slider.Thumb className="block w-5 h-5 bg-white border-2 border-blue-500 rounded-full" />
                  </Slider.Root>
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={startProcessing}
            disabled={!images.length || processing}
            className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
          >
            Start Processing
          </button>

          {processing && (
            <Progress.Root
              value={progress}
              className="h-2 bg-gray-200 rounded-full overflow-hidden mt-4"
            >
              <Progress.Indicator
                style={{ width: `${progress}%` }}
                className="h-full bg-blue-500 transition-all"
              />
            </Progress.Root>
          )}
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4">Results</h2>
          <div className="grid grid-cols-4 gap-4">
            {results.map((result, i) => (
              <div key={i} className="aspect-square bg-gray-100 rounded-lg p-4">
                <img
                  src={URL.createObjectURL(result.file)}
                  alt={`Result ${i + 1}`}
                  className="w-full h-full object-cover rounded"
                />
                <div className="mt-2 text-sm">
                  <p>Score: {result.analysis.total_score.toFixed(1)}</p>
                  <p>Blur: {result.analysis.blur_score.toFixed(0)}</p>
                  <p>Eyes: {result.analysis.eyes_open ? '✓' : '✗'}</p>
                  <p>Smile: {result.analysis.smiling ? '✓' : '✗'}</p>
                  <p>Exposure: {result.analysis.exposure_quality}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App