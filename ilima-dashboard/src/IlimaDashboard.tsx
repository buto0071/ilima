import { useState, useEffect, useRef } from 'react';
import {
    Activity,
    Server,
    Database,
    Cpu,
    Share2,
    Terminal,
    UploadCloud,
    Send,
    Box,
    Zap,
    FileCode,
    CheckCircle2,
    Lock,
    Network
} from 'lucide-react';

// --- Types ---
interface Peer {
    id: string;
    address: string;
    role: 'Coordinator' | 'Worker';
    latency: number;
    status: 'active' | 'syncing' | 'computing';
}

interface Metrics {
    cpu_usage: number;
    total_mem: number;
    used_mem: number;
    gpu_usage: number;
}

interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    thinking?: string;
    isThinking?: boolean;
    tps?: number;
    tokens?: number;
    timestamp: number;
}

const IlimaDashboard = () => {
    // --- State ---
    const [nodeId] = useState<string>('12D3Koo...8jL2');
    const [peers, setPeers] = useState<Peer[]>([]); // Start empty, wait for sync

    // Real state for selected model
    const [selectedModel, setSelectedModel] = useState<string | null>(null);
    const [availableModels, setAvailableModels] = useState<string[]>([]);

    // Derived state for the UI's "activeModel" object
    const activeModel = selectedModel ? {
        name: selectedModel,
        cid: "Qm" + btoa(selectedModel).substring(0, 16), // Mock CID based on filename
        size: "Unknown"
    } : null;

    const [uploadState, setUploadState] = useState<'idle' | 'uploading' | 'hashing' | 'complete'>('idle');
    const [uploadProgress, setUploadProgress] = useState(0);
    const [repoId, setRepoId] = useState('');
    const [modelStatus, setModelStatus] = useState<'Offline' | 'Loading' | 'Active' | 'Error'>('Offline');

    const [promptInput, setPromptInput] = useState('');
    const [isInferencing, setIsInferencing] = useState(false);
    const [inferenceStats, setInferenceStats] = useState({ tps: 0, progress: 0 });

    const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
        { id: '0', role: 'system', content: 'Ilima Cluster Ready. Load a model to begin.', timestamp: Date.now() }
    ]);

    const [logs, setLogs] = useState<string[]>([
        '[System] ilima v0.1.0 initialized',
        '[Network] mDNS service started',
    ]);

    // Optimization state
    const [, setOptimizationConfig] = useState<any>(null);
    const [, setCacheStats] = useState<any>(null);
    const [, setSchedulerStats] = useState<any>(null);

    const chatEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    const handleStopInference = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
            setIsInferencing(false);
            addLog('[RPC] Inference stopped by user.');
        }
    };

    const addLog = (msg: string) => {
        setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 50));
    };

    // Auto-scroll chat
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatHistory]);

    // Fetch models and check status on mount
    useEffect(() => {
        const fetchModelsAndStatus = async () => {
            try {
                // 1. Fetch available models
                const resModels = await fetch('http://localhost:3000/api/models');
                const dataModels = await resModels.json();
                if (dataModels.models) {
                    setAvailableModels(dataModels.models);
                }

                // 2. Check current status (for pre-loaded model)
                const resStatus = await fetch('http://localhost:3000/api/status');
                const dataStatus = await resStatus.json();

                if (dataStatus.model_status) {
                    setModelStatus(dataStatus.model_status);
                }

                if (dataStatus.active_model) {
                    console.log("Found pre-loaded model:", dataStatus.active_model);
                    setSelectedModel(dataStatus.active_model);
                    setChatHistory(prev => [
                        ...prev,
                        {
                            id: 'init-loaded',
                            role: 'system',
                            content: `Model "${dataStatus.active_model}" loaded successfully (Pre-loaded).`,
                            timestamp: Date.now()
                        }
                    ]);
                }

            } catch (e) {
                console.error("Failed to fetch initial state", e);
                addLog(`[Error] Failed to fetch initial state: ${e}`);
            }
        };
        fetchModelsAndStatus();
    }, []);

    const [metrics, setMetrics] = useState<Metrics | null>(null);
    const [slotActivity, setSlotActivity] = useState<{ isProcessing: boolean; decoded: number; remain: number } | null>(null);

    // Poll for Peers & Metrics
    useEffect(() => {
        let lastLoggedStatus = ''; // Track to prevent log flooding

        const fetchPeers = async () => {
            try {
                const res = await fetch('http://localhost:3000/api/peers');
                const data = await res.json();
                if (data.peers) {
                    setPeers(data.peers);
                }
                if (data.metrics) {
                    setMetrics(data.metrics);
                }
            } catch (e) {
                // Silent fail
            }
        };

        const fetchStatus = async () => {
            try {
                const res = await fetch('http://localhost:3000/api/status');
                const data = await res.json();
                if (data.active_model) setSelectedModel(data.active_model);

                if (data.model_status) {
                    if (typeof data.model_status === 'object') {
                        if (data.model_status.Error) {
                            setModelStatus('Error');
                            if (lastLoggedStatus !== data.model_status.Error) {
                                addLog(`[Status] Error: ${data.model_status.Error}`);
                                lastLoggedStatus = data.model_status.Error;
                            }
                        } else {
                            setModelStatus('Offline');
                        }
                    } else {
                        setModelStatus(data.model_status);
                        if (data.model_status !== 'Error') {
                            lastLoggedStatus = '';
                        }
                    }
                }
            } catch (e) { }
        };

        const fetchSlots = async () => {
            try {
                const res = await fetch('http://localhost:8081/slots');
                const slots = await res.json();
                const s = Array.isArray(slots) ? slots[0] : slots?.value?.[0];
                if (s && s.is_processing) {
                    const nt = Array.isArray(s.next_token) ? s.next_token[0] : s.next_token;
                    setSlotActivity({
                        isProcessing: true,
                        decoded: nt?.n_decoded ?? 0,
                        remain: nt?.n_remain ?? 0
                    });
                } else {
                    setSlotActivity(null);
                }
            } catch (e) {
                setSlotActivity(null);
            }
        };

        fetchPeers();
        fetchStatus();
        fetchSlots();
        const interval = setInterval(() => {
            fetchPeers();
            fetchStatus();
            fetchSlots();
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    // Poll for Optimization Stats
    useEffect(() => {
        const fetchOptimizationData = async () => {
            try {
                // Fetch optimization config
                const resConfig = await fetch('http://localhost:3000/api/optimization/config');
                const dataConfig = await resConfig.json();
                if (dataConfig.config) setOptimizationConfig(dataConfig.config);

                // Fetch cache stats
                const resCache = await fetch('http://localhost:3000/api/optimization/stats');
                const dataCache = await resCache.json();
                if (dataCache.cache) setCacheStats(dataCache.cache);

                // Fetch scheduler stats
                const resScheduler = await fetch('http://localhost:3000/api/scheduler/stats');
                const dataScheduler = await resScheduler.json();
                setSchedulerStats(dataScheduler);
            } catch (e) {
                // Silent fail
            }
        };

        fetchOptimizationData();
        const interval = setInterval(fetchOptimizationData, 3000);
        return () => clearInterval(interval);
    }, []);


    // --- Actions ---

    // --- Actions ---

    const loadModel = async (modelName: string) => {
        if (!modelName) return;
        addLog(`[System] Requesting to load model: ${modelName}...`);
        setModelStatus('Loading');

        try {
            const res = await fetch('http://localhost:3000/api/models/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_path: modelName })
            });
            const data = await res.json();
            if (data.status === 'loading') {
                addLog(`[System] Load initiated. Waiting for server...`);
            } else if (data.status === 'already_loaded') {
                addLog(`[System] Model already loaded. No reload needed.`);
            } else {
                addLog(`[Error] Load failed: ${JSON.stringify(data)}`);
                setModelStatus('Error');
            }
        } catch (e) {
            addLog(`[Error] Failed to call load API: ${e}`);
            setModelStatus('Error');
        }
    };

    const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setUploadState('uploading');
        setUploadProgress(10);
        addLog(`[CAS] Initiating upload stream for "${file.name}"...`);

        const formData = new FormData();
        if (repoId.trim()) {
            formData.append('repo_id', repoId.trim());
        }
        formData.append('model', file);

        try {
            // Simulate progress for visual feedback
            const progressInterval = setInterval(() => {
                setUploadProgress(prev => Math.min(prev + 10, 90));
            }, 200);

            const response = await fetch('http://localhost:3000/api/upload', {
                method: 'POST',
                body: formData,
            });

            clearInterval(progressInterval);
            const data = await response.json();

            if (data.status === 'uploaded') {
                setUploadProgress(100);
                setUploadState('hashing'); // Visual step
                setTimeout(() => {
                    setUploadState('complete');
                    setSelectedModel(data.filename);
                    setAvailableModels(prev => [...prev, data.filename]);
                    addLog(`[CAS] Upload Complete. Indexed as ${data.filename}`);
                    setChatHistory(prev => [...prev, {
                        id: Date.now().toString(),
                        role: 'system',
                        content: `Model "${file.name}" loaded successfully.`,
                        timestamp: Date.now()
                    }]);
                    // Reset upload state after a delay
                    setTimeout(() => setUploadState('idle'), 2000);
                }, 800);

                // Auto-load after upload
                setTimeout(() => {
                    loadModel(data.filename);
                }, 1000);

            } else {
                setUploadState('idle');
                addLog(`[Error] Upload failed: ${data.error}`);
            }
        } catch (e) {
            setUploadState('idle');
            addLog(`[Error] Network error during upload: ${e}`);
        }
    };

    const handleSendPrompt = async () => {
        if (!promptInput.trim() || !activeModel) return;

        const newPrompt = promptInput;
        setPromptInput('');
        setIsInferencing(true);
        setInferenceStats({ tps: 0, progress: 15 });

        const userMsgId = Date.now().toString();
        const assistantMsgId = (Date.now() + 1).toString();

        // Add User Message + Live Assistant Placeholder
        setChatHistory(prev => [
            ...prev,
            {
                id: userMsgId,
                role: 'user',
                content: newPrompt,
                timestamp: Date.now()
            },
            {
                id: assistantMsgId,
                role: 'assistant',
                content: '',
                thinking: '',
                isThinking: true,
                tokens: 0,
                tps: 0,
                timestamp: Date.now()
            }
        ]);

        addLog(`[RPC] Streaming chat completion from swarm...`);

        const workerPeer = peers.length > 0 ? peers[0] : null;
        if (workerPeer) {
            addLog(`[Scheduler] Task assigned to Node ${workerPeer.id} (${workerPeer.address})`);
            setPeers(prev => prev.map(p => p.id === workerPeer.id ? { ...p, status: 'computing' } : p));
        }

        const controller = new AbortController();
        abortControllerRef.current = controller;
        const startTime = performance.now();

        try {
            const response = await fetch('http://localhost:3000/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: controller.signal,
                body: JSON.stringify({
                    model: selectedModel || 'default',
                    messages: [
                        {
                            role: 'system',
                            content: 'You are a helpful assistant. Keep any internal reasoning concise and directly provide the complete, detailed final answer.'
                        },
                        { role: 'user', content: newPrompt }
                    ],
                    stream: true,
                    temperature: 0.7,
                    cache_prompt: true
                })
            });

            if (!response.ok || !response.body) {
                throw new Error(`Server returned ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let accThinking = '';
            let accContent = '';
            let tokenCount = 0;
            let reportedTps = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const rawLine of lines) {
                    const line = rawLine.trim();
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();
                    if (payload === '[DONE]') continue;

                    try {
                        const parsed = JSON.parse(payload);
                        const delta = parsed.choices?.[0]?.delta;
                        if (delta) {
                            if (delta.reasoning_content) {
                                accThinking += delta.reasoning_content;
                                tokenCount += 1;
                            }
                            if (delta.content) {
                                accContent += delta.content;
                                tokenCount += 1;
                            }
                        }
                        if (parsed.timings?.predicted_per_second) {
                            reportedTps = Number(parsed.timings.predicted_per_second.toFixed(1));
                        }
                    } catch {
                        // ignore partial JSON line
                    }
                }

                const elapsedSec = Math.max((performance.now() - startTime) / 1000, 0.1);
                const liveTps = reportedTps || Number((tokenCount / elapsedSec).toFixed(1));
                const stillThinking = accContent.length === 0;

                setInferenceStats({
                    tps: liveTps,
                    progress: Math.min(15 + Math.floor(tokenCount / 10), 95)
                });

                setChatHistory(prev => prev.map(m =>
                    m.id === assistantMsgId
                        ? {
                            ...m,
                            thinking: accThinking,
                            content: accContent,
                            isThinking: stillThinking,
                            tokens: tokenCount,
                            tps: liveTps
                        }
                        : m
                ));
            }

            const totalSec = Math.max((performance.now() - startTime) / 1000, 0.1);
            const finalTps = reportedTps || Number((tokenCount / totalSec).toFixed(1));
            setInferenceStats({ tps: finalTps, progress: 100 });
            addLog(`[RPC] Stream complete (${tokenCount} tokens @ ${finalTps} tok/s).`);

            setChatHistory(prev => prev.map(m =>
                m.id === assistantMsgId
                    ? {
                        ...m,
                        thinking: accThinking,
                        content: accContent || (accThinking ? '(Stopped during thinking phase)' : '(No output generated)'),
                        isThinking: false,
                        tokens: tokenCount,
                        tps: finalTps
                    }
                    : m
            ));
        } catch (e: any) {
            if (e?.name === 'AbortError') {
                setChatHistory(prev => prev.map(m =>
                    m.id === assistantMsgId
                        ? { ...m, isThinking: false, content: m.content || '(Generation stopped by user)' }
                        : m
                ));
            } else {
                addLog(`[Error] Stream error: ${e}`);
                setChatHistory(prev => prev.map(m =>
                    m.id === assistantMsgId
                        ? { ...m, isThinking: false, content: `Error: ${e}` }
                        : m
                ));
            }
        } finally {
            abortControllerRef.current = null;
            setIsInferencing(false);
            if (workerPeer) {
                setPeers(prev => prev.map(p => p.id === workerPeer.id ? { ...p, status: 'active' } : p));
            }
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-6 flex flex-col">

            {/* Header */}
            <header className="flex justify-between items-center mb-6 border-b border-slate-800 pb-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/20">
                        <Share2 className="text-white w-6 h-6" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-white tracking-tight">Ilima Control Plane</h1>
                        <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            ID: {nodeId} • Role: COORDINATOR
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-4 bg-slate-900 px-4 py-2 rounded-lg border border-slate-800">
                    <Network className="w-4 h-4 text-indigo-400" />
                    <span className="text-xs font-mono text-indigo-300">DHT Network: SYNCED</span>
                </div>

                {/* System Metrics */}
                {metrics && (
                    <div className="flex items-center gap-4 bg-slate-900 px-4 py-2 rounded-lg border border-slate-800 ml-4">
                        <div className="flex items-center gap-2">
                            <Cpu className="w-4 h-4 text-emerald-400" />
                            <span className="text-xs font-mono text-emerald-300">CPU: {metrics.cpu_usage.toFixed(1)}%</span>
                        </div>
                        <div className="w-[1px] h-3 bg-slate-700"></div>
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-purple-400" />
                            <span className="text-xs font-mono text-purple-300">RAM: {(metrics.used_mem / 1024 / 1024 / 1024).toFixed(1)} GB</span>
                        </div>
                    </div>
                )}
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">

                {/* Left Column: Model Management (4 cols) */}
                <div className="lg:col-span-4 space-y-6">

                    {/* Upload Zone */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
                        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Database className="w-4 h-4" /> CAS Model Loader
                        </h2>

                        {/* P2P Tokenizer Config */}
                        <div className="mb-4">
                            <label className="text-[10px] uppercase text-slate-500 font-bold mb-1 block">HuggingFace Repo ID (Optional)</label>
                            <input
                                type="text"
                                placeholder="e.g. mistralai/Mistral-7B-v0.1"
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 mb-1"
                                value={repoId}
                                onChange={(e) => setRepoId(e.target.value)}
                            />
                            <p className="text-[10px] text-slate-500">Auto-downloads tokenizer.json for deployed agents.</p>
                        </div>

                        {!activeModel ? (
                            <div
                                onClick={() => uploadState === 'idle' && fileInputRef.current?.click()}
                                className={`
                  border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center transition-all cursor-pointer
                  ${uploadState === 'idle'
                                        ? 'border-slate-700 hover:border-indigo-500 hover:bg-slate-800/50'
                                        : 'border-indigo-500/50 bg-indigo-500/5'}
                `}
                            >
                                {uploadState === 'idle' && (
                                    <>
                                        <UploadCloud className="w-10 h-10 text-indigo-400 mb-3" />
                                        <h3 className="text-sm font-medium text-white">Click to Upload GGUF</h3>
                                        <p className="text-xs text-slate-500 mt-1">Simulates file hashing & DHT announcement</p>
                                    </>
                                )}

                                {(uploadState === 'uploading' || uploadState === 'hashing') && (
                                    <div className="w-full">
                                        <div className="flex justify-between text-xs mb-2">
                                            <span className="text-indigo-300">
                                                {uploadState === 'uploading' ? 'Streaming chunks...' : 'Calculating SHA-256...'}
                                            </span>
                                            <span className="text-white font-mono">{uploadProgress}%</span>
                                        </div>
                                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-indigo-500 transition-all duration-300 ease-out"
                                                style={{ width: `${uploadProgress}%` }}
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4">
                                <div className="flex items-start gap-3">
                                    <div className="p-2 bg-emerald-500/20 rounded-lg">
                                        <Box className="w-5 h-5 text-emerald-400" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <h3 className="text-sm font-bold text-emerald-100 truncate">{activeModel.name}</h3>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded font-mono border border-slate-700">
                                                {activeModel.size}
                                            </span>
                                            <span className="text-[10px] text-emerald-400/80 font-mono truncate">
                                                CID: {activeModel.cid}
                                            </span>
                                        </div>
                                    </div>
                                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                </div>
                            </div>
                        )}

                        <div className="mt-4 pt-4 border-t border-slate-800">
                            <div className="flex items-center gap-2 text-xs text-slate-500">
                                <Lock className="w-3 h-3" />
                                <span>Files are encrypted & content-addressed</span>
                            </div>
                        </div>
                    </div>

                    {/* Node Status */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex-1">
                        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Server className="w-4 h-4" /> Available Compute
                        </h2>
                        <div className="space-y-3">
                            {peers.length === 0 ? (
                                <div className="p-4 text-center text-slate-600 text-xs italic border border-slate-800 rounded-lg border-dashed">
                                    Waiting for swarm peers...
                                </div>
                            ) : (
                                peers.map(peer => (
                                    <div key={peer.id} className="flex items-center justify-between p-3 bg-slate-950 rounded-lg border border-slate-800">
                                        <div className="flex items-center gap-3">
                                            <div className={`w-2 h-2 rounded-full ${peer.status === 'computing' ? 'bg-indigo-400 animate-pulse' : 'bg-emerald-500'}`} />
                                            <div>
                                                <div className="text-sm font-medium text-slate-300 max-w-[150px] truncate" title={peer.id}>{peer.id}</div>
                                                <div className="text-[10px] text-slate-500 font-mono">{peer.address}</div>
                                            </div>
                                        </div>
                                        {peer.status === 'computing' && (
                                            <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-2 py-1 rounded animate-pulse">
                                                Running Inference...
                                            </span>
                                        )}
                                    </div>
                                )))}
                        </div>
                    </div>

                </div>

                {/* Center Column: Inference Chat (5 cols) */}
                <div className="lg:col-span-5 flex flex-col bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg relative">

                    {/* Top Bar with Model Dropdown and Upload Button */}
                    <div className="p-3 bg-slate-900/80 backdrop-blur border-b border-slate-800 flex justify-between items-center z-10">
                        <div className="flex items-center gap-2">
                            <Zap className="w-4 h-4 text-yellow-500" />
                            <span className="text-sm font-medium text-slate-300">Remote Inference Stream</span>
                            {slotActivity?.isProcessing && (
                                <span className="text-[10px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded flex items-center gap-1.5 animate-pulse">
                                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping" />
                                    Decoding: {slotActivity.decoded.toLocaleString()} tok
                                </span>
                            )}
                        </div>

                        <div className="flex items-center gap-2">
                            <select
                                className="bg-slate-950 border border-slate-800 text-emerald-400 text-[10px] rounded px-2 py-1 focus:outline-none focus:border-indigo-500 max-w-[150px]"
                                value={selectedModel || ""}
                                onChange={(e) => {
                                    const m = e.target.value;
                                    if (m) loadModel(m);
                                }}
                                disabled={modelStatus === 'Loading'}
                            >
                                <option value="">Select Model...</option>
                                {availableModels.map(m => (
                                    <option key={m} value={m}>{m}</option>
                                ))}
                            </select>

                            {/* Hidden input for the upload functionality */}
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleFileSelect}
                                className="hidden"
                                accept=".gguf,.bin"
                            />

                            <button
                                onClick={() => fileInputRef.current?.click()}
                                disabled={uploadState !== 'idle'}
                                className="bg-indigo-600 hover:bg-indigo-500 text-white rounded p-1"
                                title="Upload New Model"
                            >
                                <UploadCloud className="w-3 h-3" />
                            </button>
                        </div>
                    </div>

                    {/* Chat History */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-950/50">
                        {chatHistory.map((msg) => {
                            // Also extract <think>...</think> if present inside msg.content from non-streaming calls
                            let displayThinking = msg.thinking || '';
                            let displayContent = msg.content || '';
                            if (!displayThinking && displayContent.startsWith('<think>')) {
                                const endIdx = displayContent.indexOf('</think>');
                                if (endIdx !== -1) {
                                    displayThinking = displayContent.slice(7, endIdx).trim();
                                    displayContent = displayContent.slice(endIdx + 8).trim();
                                } else {
                                    displayThinking = displayContent.slice(7).trim();
                                    displayContent = '';
                                }
                            }

                            return (
                                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`
                  max-w-[90%] rounded-lg p-3 text-sm leading-relaxed
                  ${msg.role === 'user'
                                            ? 'bg-indigo-600 text-white rounded-br-none'
                                            : msg.role === 'system'
                                                ? 'bg-slate-800/50 text-slate-400 text-xs border border-slate-800 w-full max-w-full text-center py-2'
                                                : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-bl-none w-full'}
                `}>
                                        {msg.role === 'assistant' && (
                                            <div className="flex items-center justify-between mb-2">
                                                <div className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider flex items-center gap-1">
                                                    <FileCode className="w-3 h-3" /> Generated Output
                                                </div>
                                                {(msg.tps || msg.tokens) ? (
                                                    <div className="text-[10px] font-mono text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">
                                                        {msg.tokens ? `${msg.tokens} tok` : ''}{msg.tps ? ` • ${msg.tps} tok/s` : ''}
                                                    </div>
                                                ) : null}
                                            </div>
                                        )}

                                        {msg.role === 'assistant' && (displayThinking || msg.isThinking) && (
                                            <details open className="mb-3 bg-slate-950/80 border border-amber-500/30 rounded-lg overflow-hidden">
                                                <summary className="px-3 py-1.5 bg-amber-500/10 text-amber-300 text-xs font-mono cursor-pointer flex items-center justify-between select-none">
                                                    <span className="flex items-center gap-2">
                                                        {msg.isThinking && (
                                                            <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                                                        )}
                                                        {msg.isThinking ? 'Thinking live...' : 'Thought Process (click to toggle)'}
                                                    </span>
                                                    <span className="text-[10px] text-amber-400/70">
                                                        {displayThinking.length} chars
                                                    </span>
                                                </summary>
                                                <div className="p-3 text-xs font-mono text-amber-100/80 whitespace-pre-wrap max-h-64 overflow-y-auto leading-relaxed border-t border-amber-500/20">
                                                    {displayThinking || 'Initializing reasoning...'}
                                                </div>
                                            </details>
                                        )}

                                        <div className="whitespace-pre-wrap break-words">
                                            {displayContent || (msg.isThinking ? '' : msg.content)}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                        <div ref={chatEndRef} />
                    </div>

                    {/* Progress Bar Overlay */}
                    {isInferencing && (
                        <div className="h-1 bg-slate-800 w-full">
                            <div className="h-full bg-indigo-500 transition-all duration-200" style={{ width: `${inferenceStats.progress}%` }} />
                        </div>
                    )}

                    {/* Loading Overlay */}
                    {modelStatus === 'Loading' && (
                        <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                            <div className="w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                            <div className="text-indigo-400 font-bold animate-pulse">Loading Model to VRAM...</div>
                            <div className="text-xs text-slate-500 mt-2">Allocating buffers on worker nodes</div>
                        </div>
                    )}

                    {/* Input Area */}
                    <div className="p-4 bg-slate-900 border-t border-slate-800">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={promptInput}
                                onChange={(e) => setPromptInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSendPrompt()}
                                placeholder={
                                    modelStatus === 'Loading' ? "Model is loading..." :
                                        isInferencing ? "Streaming thoughts & response..." :
                                            modelStatus === 'Active' ? "Send prompt to swarm..." :
                                                "Select a model to begin"
                                }
                                disabled={modelStatus !== 'Active' || isInferencing}
                                className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            />
                            {isInferencing ? (
                                <button
                                    onClick={handleStopInference}
                                    className="bg-rose-600 hover:bg-rose-500 text-white rounded-lg px-4 py-2 text-xs font-bold transition-colors flex items-center justify-center"
                                    title="Stop Generation"
                                >
                                    Stop
                                </button>
                            ) : (
                                <button
                                    onClick={handleSendPrompt}
                                    disabled={modelStatus !== 'Active' || !promptInput.trim() || isInferencing}
                                    className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white rounded-lg px-4 py-2 transition-colors flex items-center justify-center"
                                >
                                    <Send className="w-4 h-4" />
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right Column: Logs (3 cols) */}
                <div className="lg:col-span-3 flex flex-col bg-black border border-slate-800 rounded-xl overflow-hidden shadow-sm">
                    <div className="flex items-center gap-2 text-slate-500 p-3 border-b border-slate-900 bg-slate-950">
                        <Terminal className="w-3 h-3" />
                        <span className="text-xs font-bold uppercase">Event Log</span>
                    </div>
                    <div className="flex-1 overflow-y-auto p-3 space-y-2 font-mono text-[10px]">
                        {logs.map((log, i) => (
                            <div key={i} className="text-slate-400 break-all leading-relaxed border-l-2 border-slate-800 pl-2 hover:bg-slate-900/30 transition-colors">
                                <span className="text-indigo-500 opacity-70 mr-2">➜</span>
                                {log}
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );
};

export default IlimaDashboard;
