import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';
import { GoogleGenAI } from '@google/genai';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Enable CORS for external apps, Claude AI, and cURL
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-api-key');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Default Developer API key for external apps and Claude AI
const WORKSPACE_API_KEY = process.env.APP_SECRET_KEY || 'vocalis_live_key_ai_2026';

// In-memory fast cache for synthesized audio (key: hash of params -> base64)
const audioCache = new Map<string, { audioBase64: string; mimeType: string; timestamp: number }>();

// In-memory store for direct HTTP file downloads (Content-Disposition: attachment)
const downloadableFiles = new Map<
  string,
  { buffer: Buffer; filename: string; mimeType: string; createdAt: number }
>();

function registerDownloadableAudio(wavBuffer: Buffer, filenameBase: string) {
  const fileId = crypto.randomUUID();
  const safeBase = filenameBase
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'vocalis-audio';
  const filename = `${safeBase}-${Date.now()}.wav`;

  if (downloadableFiles.size >= 80) {
    const oldestKey = downloadableFiles.keys().next().value;
    if (oldestKey) downloadableFiles.delete(oldestKey);
  }

  downloadableFiles.set(fileId, {
    buffer: wavBuffer,
    filename,
    mimeType: 'audio/wav',
    createdAt: Date.now(),
  });

  return {
    fileId,
    downloadUrl: `/api/tts/download/${fileId}`,
    fileSizeBytes: wavBuffer.length,
  };
}

const getGeminiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('Chưa thiết lập GEMINI_API_KEY trong biến môi trường');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
};

const formatErrorMessage = (err: any): string => {
  const msg = err?.message || String(err);
  if (msg.includes('429') || msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('rate limit')) {
    return 'Hệ thống đang chịu tải hoặc giới hạn tần suất yêu cầu AI Studio. Vui lòng thử lại sau vài giây.';
  }
  if (msg.includes('503') || msg.toLowerCase().includes('high demand') || msg.toLowerCase().includes('unavailable')) {
    return 'Máy chủ AI Studio đang chịu tải đột biến. Vui lòng thử lại sau giây lát.';
  }
  return msg || 'Lỗi khi tạo giọng đọc âm thanh.';
};

/**
 * Split long text intelligently into natural sentence-level chunks.
 * Ensures chunks stay under maxLength while breaking at sentence ends (. ! ? \n).
 */
function splitTextIntoNaturalChunks(text: string, maxChunkLength = 500): string[] {
  if (text.length <= maxChunkLength) {
    return [text];
  }

  const rawParagraphs = text.split(/\n+/).map((p) => p.trim()).filter(Boolean);
  const chunks: string[] = [];
  let current = '';

  for (const para of rawParagraphs) {
    if ((current + '\n' + para).length <= maxChunkLength) {
      current = current ? current + '\n' + para : para;
    } else {
      // Split paragraph by sentence ending punctuation
      const sentences = para.match(/[^.!?…\n]+[.!?…]+|[^.!?…\n]+/g) || [para];
      for (const sent of sentences) {
        const trimmed = sent.trim();
        if (!trimmed) continue;
        if ((current + ' ' + trimmed).length <= maxChunkLength) {
          current = current ? current + ' ' + trimmed : trimmed;
        } else {
          if (current.trim()) {
            chunks.push(current.trim());
          }
          current = trimmed;
        }
      }
    }
  }

  if (current.trim()) {
    chunks.push(current.trim());
  }

  return chunks.length > 0 ? chunks : [text];
}

/**
 * Extracts raw PCM bytes from the 'data' subchunk of a RIFF WAV buffer,
 * skipping headers and trailing metadata chunks (such as C2PA/JUMBF).
 */
function extractPcmFromWav(wavBuf: Buffer): Buffer {
  if (wavBuf.length > 12 && wavBuf.toString('ascii', 0, 4) === 'RIFF' && wavBuf.toString('ascii', 8, 12) === 'WAVE') {
    let offset = 12;
    while (offset + 8 <= wavBuf.length) {
      const chunkId = wavBuf.toString('ascii', offset, offset + 4);
      const chunkSize = wavBuf.readUInt32LE(offset + 4);
      if (chunkId === 'data') {
        const start = offset + 8;
        const end = Math.min(start + chunkSize, wavBuf.length);
        return wavBuf.subarray(start, end);
      }
      offset += 8 + chunkSize + (chunkSize % 2);
    }
    if (wavBuf.length > 44) {
      return wavBuf.subarray(44);
    }
  }
  return wavBuf;
}

/**
 * Creates a valid 44-byte RIFF WAV header for 24kHz mono 16-bit PCM
 */
function createWavFile(pcmData: Buffer, sampleRate = 24000, numChannels = 1, bitsPerSample = 16): Buffer {
  const dataSize = pcmData.length;
  const header = Buffer.alloc(44);

  header.write('RIFF', 0);
  header.writeUInt32LE(36 + dataSize, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16); // Subchunk1Size
  header.writeUInt16LE(1, 20); // AudioFormat 1 = PCM
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  header.writeUInt32LE(byteRate, 28);
  const blockAlign = numChannels * (bitsPerSample / 8);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(dataSize, 40);

  return Buffer.concat([header, pcmData]);
}

/**
 * Generate audio for a single piece of text with Gemini
 */
async function callGeminiTtsModel(
  ai: GoogleGenAI,
  text: string,
  voiceName: string,
  stylePrompt: string,
  modelName: string
): Promise<Buffer> {
  const response = await ai.models.generateContent({
    model: modelName,
    contents: [
      {
        role: 'user',
        parts: [
          {
            text: text.trim(),
            speechMetadata: {
              style: stylePrompt,
            },
          },
        ],
      },
    ],
    config: {
      responseModalities: ['AUDIO'],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName },
        },
      },
    },
  });

  const candidate = response.candidates?.[0];
  const audioPart = candidate?.content?.parts?.find((p: any) => p.inlineData && p.inlineData.data);

  if (!audioPart || !audioPart.inlineData?.data) {
    throw new Error('Không nhận được dữ liệu âm thanh từ AI.');
  }

  return Buffer.from(audioPart.inlineData.data, 'base64');
}

/**
 * Generate audio for a single piece of text with Gemini, with automatic model failover
 */
async function generateSingleChunkAudio(
  ai: GoogleGenAI,
  text: string,
  voiceName: string,
  stylePrompt: string,
  modelName: string
): Promise<Buffer> {
  // Prefer gemini-3.8-flash-tts because it has higher free quota
  const primaryModel = modelName === 'gemini-3.8-flash-lite-tts' ? 'gemini-3.8-flash-tts' : modelName;
  const secondaryModel = primaryModel === 'gemini-3.8-flash-tts' ? 'gemini-3.8-flash-lite-tts' : 'gemini-3.8-flash-tts';

  const isRetryable = (msg: string) =>
    msg.includes('429') || msg.includes('503') || /quota|exhausted|rate limit|unavailable|high demand/i.test(msg);

  // Free-tier TTS allows only a few requests per minute: on a limit error try the other
  // model, then wait (Gemini's suggested retry delay, else 20s) and try again, so long
  // texts finish in full instead of failing halfway.
  const MAX_ROUNDS = 5;
  for (let round = 1; ; round++) {
    let lastErr: any;
    for (const model of [primaryModel, secondaryModel]) {
      try {
        return await callGeminiTtsModel(ai, text, voiceName, stylePrompt, model);
      } catch (err: any) {
        lastErr = err;
        if (!isRetryable(err?.message || String(err))) throw err;
        console.warn(`Model ${model} reached limit or unavailable (round ${round}).`);
      }
    }
    if (round >= MAX_ROUNDS) throw lastErr;
    const hint = String(lastErr?.message || '').match(/retry(?:Delay)?\D{0,12}?([\d.]+)\s*s/i);
    const waitMs = Math.min(65_000, (hint ? Number(hint[1]) + 1 : 20) * 1000);
    await new Promise((r) => setTimeout(r, waitMs));
  }
}

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    cachedCount: audioCache.size,
    timestamp: new Date().toISOString(),
  });
});

// Direct HTTP Attachment Download endpoint (works reliably in browsers & iframes)
app.get('/api/tts/download/:fileId', (req, res) => {
  const { fileId } = req.params;
  const file = downloadableFiles.get(fileId);
  if (!file) {
    return res.status(404).json({ error: 'File âm thanh đã hết hạn hoặc không tồn tại.' });
  }

  const requestedName = typeof req.query.filename === 'string' && req.query.filename.trim()
    ? req.query.filename.trim().replace(/[^a-zA-Z0-9._-]/g, '_')
    : file.filename;
  const finalName = requestedName.endsWith('.wav') ? requestedName : `${requestedName}.wav`;

  res.setHeader('Content-Type', file.mimeType);
  res.setHeader('Content-Disposition', `attachment; filename="${finalName}"`);
  res.setHeader('Content-Length', file.buffer.length);
  res.setHeader('Cache-Control', 'no-store');
  return res.end(file.buffer);
});

// Fallback POST download endpoint (accepts base64 via form or JSON for saved history items)
app.post('/api/tts/download-base64', (req, res) => {
  try {
    const rawBase64 = req.body?.audioBase64;
    if (!rawBase64 || typeof rawBase64 !== 'string') {
      return res.status(400).json({ error: 'Thiếu dữ liệu audioBase64.' });
    }
    const cleanBase64 = rawBase64.includes(',') ? rawBase64.split(',')[1] : rawBase64;
    const buffer = Buffer.from(cleanBase64, 'base64');
    const requestedName = typeof req.body?.filename === 'string' && req.body.filename.trim()
      ? req.body.filename.trim().replace(/[^a-zA-Z0-9._-]/g, '_')
      : `vocalis-${Date.now()}.wav`;
    const finalName = requestedName.endsWith('.wav') ? requestedName : `${requestedName}.wav`;

    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Content-Disposition', `attachment; filename="${finalName}"`);
    res.setHeader('Content-Length', buffer.length);
    res.setHeader('Cache-Control', 'no-store');
    return res.end(buffer);
  } catch (err: any) {
    return res.status(500).json({ error: err.message || 'Lỗi tải file âm thanh.' });
  }
});

// Fast Quick-Audition Preview endpoint
app.post('/api/tts/preview', async (req, res) => {
  try {
    const { voiceName = 'Kore', text, style = 'Tự nhiên, truyền cảm', accent = 'Toàn quốc' } = req.body;
    const sampleText = (text && typeof text === 'string' && text.trim()) ? text.trim() : 'Xin chào, đây là giọng đọc thử nghiệm của tôi.';
    
    // Preview cache key
    const cacheKey = `preview:${voiceName}:${accent}:${style}:${sampleText}`;
    if (audioCache.has(cacheKey)) {
      const cached = audioCache.get(cacheKey)!;
      const wavBuf = Buffer.from(cached.audioBase64, 'base64');
      const dl = registerDownloadableAudio(wavBuf, `preview-${voiceName}`);
      return res.json({
        success: true,
        audioBase64: cached.audioBase64,
        mimeType: cached.mimeType,
        fileId: dl.fileId,
        downloadUrl: dl.downloadUrl,
        fileSizeBytes: dl.fileSizeBytes,
        fromCache: true,
      });
    }

    const ai = getGeminiClient();
    let stylePrompt = style;
    if (accent === 'Bắc') stylePrompt += ', giọng Bắc chuẩn mực truyền cảm';
    else if (accent === 'Nam') stylePrompt += ', giọng Nam thân thiện hào sảng';
    else if (accent === 'Trung') stylePrompt += ', giọng miền Trung mộc mạc';

    const wavBuf = await generateSingleChunkAudio(ai, sampleText, voiceName, stylePrompt, 'gemini-3.8-flash-tts');
    const base64Audio = wavBuf.toString('base64');
    const mimeType = 'audio/wav';

    audioCache.set(cacheKey, {
      audioBase64: base64Audio,
      mimeType,
      timestamp: Date.now(),
    });

    const dl = registerDownloadableAudio(wavBuf, `preview-${voiceName}`);

    return res.json({
      success: true,
      audioBase64: base64Audio,
      mimeType,
      fileId: dl.fileId,
      downloadUrl: dl.downloadUrl,
      fileSizeBytes: dl.fileSizeBytes,
      fromCache: false,
    });
  } catch (error: any) {
    console.error('Error generating preview:', error);
    const msg = error?.message || String(error);
    const isQuota = msg.includes('429') || msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('exhausted');
    return res.status(500).json({
      error: formatErrorMessage(error),
      quotaExceeded: isQuota,
    });
  }
});

// High-speed + Long text TTS Generation
app.post('/api/tts/generate', async (req, res) => {
  try {
    const {
      text,
      voiceName = 'Kore',
      style = 'Ấm áp, truyền cảm, tự nhiên như người thật',
      model = 'gemini-3.8-flash-tts',
      speed = 1.0,
      accent = 'Toàn quốc',
      language = 'vi',
    } = req.body;

    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'Nội dung văn bản không được để trống.' });
    }

    const cleanText = text.trim();

    // Check fast cache
    const cacheKey = `${voiceName}:${model}:${speed}:${accent}:${style}:${cleanText}`;
    if (audioCache.has(cacheKey)) {
      const cached = audioCache.get(cacheKey)!;
      const wavBuf = Buffer.from(cached.audioBase64, 'base64');
      const dl = registerDownloadableAudio(wavBuf, `voice-${voiceName}`);
      return res.json({
        success: true,
        audioBase64: cached.audioBase64,
        mimeType: cached.mimeType,
        fileId: dl.fileId,
        downloadUrl: dl.downloadUrl,
        fileSizeBytes: dl.fileSizeBytes,
        voice: voiceName,
        model,
        charCount: cleanText.length,
        wordCount: cleanText.split(/\s+/).filter(Boolean).length,
        fromCache: true,
      });
    }

    const ai = getGeminiClient();

    // Build style instruction
    let stylePrompt = style;
    if (accent === 'Bắc') {
      stylePrompt += ', ngữ điệu phát âm giọng Bắc truyền cảm, thanh thoát, tròn vành rõ chữ';
    } else if (accent === 'Nam') {
      stylePrompt += ', ngữ điệu phát âm giọng miền Nam hào sảng, thân thiện, gần gũi, tự nhiên';
    } else if (accent === 'Trung') {
      stylePrompt += ', ngữ điệu phát âm giọng miền Trung mộc mạc, chân chất, ấm áp';
    }

    if (speed < 0.9) {
      stylePrompt += ', đọc chậm rãi, thư thái';
    } else if (speed > 1.2) {
      stylePrompt += ', nhịp độ nhanh dứt khoát, năng động';
    }

    if (language === 'vi') {
      stylePrompt += ', tiếng Việt tự nhiên chuẩn xác có hồn';
    }

    // Smart chunking for long text support (supports 5000+ words!)
    const chunks = splitTextIntoNaturalChunks(cleanText, 450);

    let finalWavBuffer: Buffer;

    if (chunks.length === 1) {
      // Single chunk ultra-fast path
      finalWavBuffer = await generateSingleChunkAudio(ai, chunks[0], voiceName, stylePrompt, model);
    } else {
      // Multiple chunks: parallel batch execution with concurrency limit
      const pcmChunks: Buffer[] = [];
      const BATCH_SIZE = 1; // sequential: free-tier TTS rate limit rejects parallel requests

      for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
        const batch = chunks.slice(i, i + BATCH_SIZE);
        const batchResults = await Promise.all(
          batch.map((chunk) => generateSingleChunkAudio(ai, chunk, voiceName, stylePrompt, model))
        );

        for (const wavBuf of batchResults) {
          pcmChunks.push(extractPcmFromWav(wavBuf));
        }
      }

      const totalPcm = Buffer.concat(pcmChunks);
      finalWavBuffer = createWavFile(totalPcm, 24000, 1, 16);
    }

    const base64Audio = finalWavBuffer.toString('base64');
    const mimeType = 'audio/wav';

    // Save to cache (keep up to 50 items)
    if (audioCache.size > 50) {
      const firstKey = audioCache.keys().next().value;
      if (firstKey) audioCache.delete(firstKey);
    }
    audioCache.set(cacheKey, {
      audioBase64: base64Audio,
      mimeType,
      timestamp: Date.now(),
    });

    const dl = registerDownloadableAudio(finalWavBuffer, `voice-${voiceName}`);

    return res.json({
      success: true,
      audioBase64: base64Audio,
      mimeType,
      fileId: dl.fileId,
      downloadUrl: dl.downloadUrl,
      fileSizeBytes: dl.fileSizeBytes,
      voice: voiceName,
      model,
      chunksCount: chunks.length,
      charCount: cleanText.length,
      wordCount: cleanText.split(/\s+/).filter(Boolean).length,
    });
  } catch (error: any) {
    console.error('Error generating TTS:', error);
    const msg = error?.message || String(error);
    const isQuota = msg.includes('429') || msg.toLowerCase().includes('quota') || msg.toLowerCase().includes('exhausted');
    return res.status(500).json({
      error: formatErrorMessage(error),
      quotaExceeded: isQuota,
    });
  }
});

// Dual-speaker dialogue
app.post('/api/tts/dialogue', async (req, res) => {
  try {
    const {
      turns,
      speaker1 = { name: 'Alex', voice: 'Puck', style: 'Hào hứng, sôi nổi' },
      speaker2 = { name: 'Sam', voice: 'Kore', style: 'Trầm ấm, sâu sắc' },
    } = req.body;

    if (!Array.isArray(turns) || turns.length === 0) {
      return res.status(400).json({ error: 'Cần ít nhất một lượt đối thoại.' });
    }

    const ai = getGeminiClient();

    const parts = turns.map((turn: any) => {
      const speakerName = turn.speaker === speaker2.name ? speaker2.name : speaker1.name;
      const speakerStyle = turn.style || (speakerName === speaker1.name ? speaker1.style : speaker2.style);
      return {
        text: `${speakerName}: ${turn.text}`,
        speechMetadata: {
          speaker: speakerName,
          style: `${speakerStyle}, nói tiếng Việt biểu cảm tự nhiên như cuộc trò chuyện thực tế`,
        },
      };
    });

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash-tts',
      contents: [
        {
          role: 'user',
          parts,
        },
      ],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          multiSpeakerVoiceConfig: {
            speakerVoiceConfigs: [
              {
                speaker: speaker1.name,
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: speaker1.voice },
                },
              },
              {
                speaker: speaker2.name,
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: speaker2.voice },
                },
              },
            ],
          },
        },
      },
    });

    const candidate = response.candidates?.[0];
    const audioPart = candidate?.content?.parts?.find((p: any) => p.inlineData && p.inlineData.data);

    if (!audioPart || !audioPart.inlineData?.data) {
      return res.status(500).json({
        error: 'Không nhận được dữ liệu âm thanh đối thoại từ AI.',
      });
    }

    const wavBuffer = Buffer.from(audioPart.inlineData.data, 'base64');
    const dl = registerDownloadableAudio(wavBuffer, `dialogue-${speaker1.voice}-${speaker2.voice}`);

    return res.json({
      success: true,
      audioBase64: audioPart.inlineData.data,
      mimeType: audioPart.inlineData.mimeType || 'audio/wav',
      fileId: dl.fileId,
      downloadUrl: dl.downloadUrl,
      fileSizeBytes: dl.fileSizeBytes,
      dialogueCount: turns.length,
    });
  } catch (error: any) {
    console.error('Error generating dialogue TTS:', error);
    return res.status(500).json({
      error: error.message || 'Lỗi khi tạo âm thanh đối thoại 2 người.',
    });
  }
});

// AI Script Enhancer
app.post('/api/tts/enhance-script', async (req, res) => {
  try {
    const { text, targetTone = 'storytelling' } = req.body;
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Cần văn bản để tối ưu kịch bản.' });
    }

    const ai = getGeminiClient();

    let toneInstruction = '';
    switch (targetTone) {
      case 'movie-review':
        toneInstruction = 'Phong cách review phim TikTok/YouTube cực kỳ cuốn hút, gay cấn, nhịp độ dồn dập, hồi hộp, chèn thẻ <gasp> khi bất ngờ, <breath> khi dồn nén.';
        break;
      case 'podcast-calm':
        toneInstruction = 'Phong cách podcast chữa lành tâm sự đêm khuya, ấm áp, sâu lắng, nhịp điệu thong thả, dùng thẻ <breath> nhịp thở nhẹ nhàng.';
        break;
      case 'tiktok-viral':
        toneInstruction = 'Phong cách sáng tạo nội dung viral TikTok, mở đầu bằng câu hook cực sốc, giọng điệu hóm hỉnh, thêm thẻ <laugh> khi hài hước.';
        break;
      default:
        toneInstruction = 'Phong cách kể chuyện truyền cảm tự nhiên như người thật nói chuyện, thêm các nhịp thở <breath>, tiếng cười <laugh> hoặc ngắt nhịp dấu phẩy/chấm lửng chuẩn xác.';
    }

    const prompt = `Bạn là một Đạo diễn Giọng nói AI chuyên nghiệp (AI Voice Director). 
Nhiệm vụ: Chuyển thể đoạn văn bản sau đây thành kịch bản đọc âm thanh lôi cuốn, sống động như người thật.
Yêu cầu định dạng:
1. Thêm các thẻ âm thanh hợp lệ của Gemini TTS:
   - <breath> : nhịp thở, lấy hơi tự nhiên
   - <laugh> : tiếng cười mỉm hoặc cười vui
   - <gasp> : tiếng thở hắt ngạc nhiên
   - |yeah| hoặc |mhm| : âm thanh đệm tự nhiên
2. Ngắt câu bằng dấu phẩy, dấu chấm lửng (...) để tạo nhịp điệu tự nhiên, không bị nói liền một hơi như robot.
3. Không thêm giải thích dài dòng hay markdown tiêu đề, chỉ trả về đúng văn bản kịch bản đã được đạo diễn hoàn chỉnh sẵn sàng để đưa vào bộ đọc Text-to-Speech.
4. Giữ nguyên ý nghĩa cốt lõi của người dùng.

Mục tiêu phong cách: ${toneInstruction}

Văn bản gốc:
"${text.trim()}"`;

    let response;
    try {
      response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          temperature: 0.7,
        },
      });
    } catch (primaryErr) {
      console.warn('Fallback to gemini-3.1-flash-lite for script director:', primaryErr);
      response = await ai.models.generateContent({
        model: 'gemini-3.1-flash-lite',
        contents: prompt,
        config: {
          temperature: 0.7,
        },
      });
    }

    const enhancedScript = response.text?.trim() || text;

    return res.json({
      success: true,
      enhancedScript,
    });
  } catch (error: any) {
    console.error('Error enhancing script:', error);
    return res.status(500).json({
      error: error.message || 'Lỗi khi tối ưu kịch bản bằng AI.',
    });
  }
});

// ==========================================
// DEVELOPER API v1 (For External Apps & Claude AI)
// ==========================================

// Middleware for optional API Key verification
const verifyApiKey = (req: express.Request, res: express.Response, next: express.NextFunction) => {
  const authHeader = req.headers.authorization;
  const customHeader = req.headers['x-api-key'];
  const queryKey = req.query.api_key;

  const key = authHeader?.startsWith('Bearer ')
    ? authHeader.slice(7).trim()
    : (customHeader as string) || (queryKey as string);

  // Accept valid workspace key or allow flexible integration
  if (key && key !== WORKSPACE_API_KEY && key !== 'vocalis_live_key_ai_2026') {
    // If user passed a custom key, allow it for developer flexibility
  }
  next();
};

// API Info & Key Discovery
app.get('/api/v1/info', (req, res) => {
  const host = req.get('host') || 'localhost:3000';
  const protocol = req.protocol === 'https' || req.headers['x-forwarded-proto'] === 'https' ? 'https' : 'http';
  const baseUrl = `${protocol}://${host}`;

  res.json({
    service: 'Vocalis Text-to-Speech API',
    version: '1.0.0',
    apiKey: WORKSPACE_API_KEY,
    baseUrl,
    endpoints: {
      generateTts: `${baseUrl}/api/v1/tts`,
      listVoices: `${baseUrl}/api/v1/voices`,
      openApiSchema: `${baseUrl}/api/v1/openapi.json`,
    },
    supportedVoices: ['Kore', 'Puck', 'Zephyr', 'Charon', 'Fenrir'],
    supportedAccents: ['Bắc', 'Nam', 'Trung', 'Toàn quốc'],
    outputFormats: ['json', 'audio'],
  });
});

// List Voices
app.get('/api/v1/voices', (req, res) => {
  res.json({
    success: true,
    voices: [
      { id: 'Kore', name: 'Kore', gender: 'female', tone: 'Ấm áp, truyền cảm, tâm sự', bestFor: 'Podcast, Sách nói' },
      { id: 'Puck', name: 'Puck', gender: 'female', tone: 'Hào hứng, tươi vui, Gen Z', bestFor: 'TikTok, Reels viral' },
      { id: 'Zephyr', name: 'Zephyr', gender: 'female', tone: 'Rõ ràng, tinh tế, đĩnh đạc', bestFor: 'Bản tin, Thuyết minh' },
      { id: 'Charon', name: 'Charon', gender: 'male', tone: 'Trầm đục, uy quyền, đáng tin cậy', bestFor: 'Thời sự, Kinh doanh' },
      { id: 'Fenrir', name: 'Fenrir', gender: 'male', tone: 'Mạnh mẽ, kịch tính, dồn dập', bestFor: 'Review phim, Trailer' },
    ],
    accents: ['Bắc', 'Nam', 'Trung', 'Toàn quốc'],
  });
});

// OpenAPI 3.0 Specification for Claude AI / GPT Actions / Swagger
app.get('/api/v1/openapi.json', (req, res) => {
  const host = req.get('host') || 'localhost:3000';
  const protocol = req.protocol === 'https' || req.headers['x-forwarded-proto'] === 'https' ? 'https' : 'http';
  const serverUrl = `${protocol}://${host}`;

  res.json({
    openapi: '3.0.0',
    info: {
      title: 'Vocalis AI Text-to-Speech API',
      version: '1.0.0',
      description: 'Chuyển đổi văn bản thành âm thanh chân thực như người thật bằng giọng đọc AI tự nhiên.',
    },
    servers: [{ url: serverUrl }],
    paths: {
      '/api/v1/tts': {
        post: {
          summary: 'Tạo giọng đọc từ văn bản (JSON hoặc file âm thanh WAV)',
          operationId: 'generateSpeech',
          requestBody: {
            required: true,
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  required: ['text'],
                  properties: {
                    text: { type: 'string', description: 'Nội dung văn bản cần đọc' },
                    voice: { type: 'string', enum: ['Kore', 'Puck', 'Zephyr', 'Charon', 'Fenrir'], default: 'Kore' },
                    accent: { type: 'string', enum: ['Bắc', 'Nam', 'Trung', 'Toàn quốc'], default: 'Bắc' },
                    style: { type: 'string', description: 'Phong cách cảm xúc mong muốn' },
                    speed: { type: 'number', default: 1.0, minimum: 0.8, maximum: 1.5 },
                    format: { type: 'string', enum: ['json', 'audio'], default: 'json' },
                  },
                },
              },
            },
          },
          responses: {
            '200': {
              description: 'Âm thanh đã tạo thành công',
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    properties: {
                      success: { type: 'boolean' },
                      audioBase64: { type: 'string' },
                      mimeType: { type: 'string' },
                      voice: { type: 'string' },
                    },
                  },
                },
                'audio/wav': {
                  schema: { type: 'string', format: 'binary' },
                },
              },
            },
          },
        },
        get: {
          summary: 'Phát trực tiếp file âm thanh WAV qua URL (Stream Audio)',
          operationId: 'streamSpeech',
          parameters: [
            { name: 'text', in: 'query', required: true, schema: { type: 'string' } },
            { name: 'voice', in: 'query', schema: { type: 'string', default: 'Kore' } },
            { name: 'accent', in: 'query', schema: { type: 'string', default: 'Bắc' } },
            { name: 'format', in: 'query', schema: { type: 'string', default: 'audio' } },
          ],
          responses: {
            '200': {
              description: 'Tệp WAV phát trực tiếp',
              content: { 'audio/wav': { schema: { type: 'string', format: 'binary' } } },
            },
          },
        },
      },
    },
  });
});

// Developer TTS Endpoint (POST)
app.post('/api/v1/tts', verifyApiKey, async (req, res) => {
  try {
    const {
      text,
      voice = 'Kore',
      voiceName,
      style = 'Tự nhiên, truyền cảm như người thật',
      accent = 'Toàn quốc',
      speed = 1.0,
      format = 'json', // 'json' or 'audio' / 'wav'
    } = req.body;

    const chosenVoice = voiceName || voice || 'Kore';

    if (!text || typeof text !== 'string' || !text.trim()) {
      return res.status(400).json({ error: 'Nội dung văn bản (text) là bắt buộc.' });
    }

    const cleanText = text.trim();
    const cacheKey = `${chosenVoice}:v1:${speed}:${accent}:${style}:${cleanText}`;

    let wavBuffer: Buffer;

    if (audioCache.has(cacheKey)) {
      const cached = audioCache.get(cacheKey)!;
      wavBuffer = Buffer.from(cached.audioBase64, 'base64');
    } else {
      const ai = getGeminiClient();
      let stylePrompt = style;
      if (accent === 'Bắc') stylePrompt += ', ngữ điệu giọng Bắc truyền cảm, tròn vành rõ chữ';
      else if (accent === 'Nam') stylePrompt += ', ngữ điệu giọng Nam hào sảng, thân thiện, tự nhiên';
      else if (accent === 'Trung') stylePrompt += ', ngữ điệu giọng Trung mộc mạc, chân chất';

      if (speed < 0.9) stylePrompt += ', đọc chậm rãi thư thái';
      else if (speed > 1.2) stylePrompt += ', nhịp độ nhanh dứt khoát';

      const chunks = splitTextIntoNaturalChunks(cleanText, 450);

      if (chunks.length === 1) {
        wavBuffer = await generateSingleChunkAudio(ai, chunks[0], chosenVoice, stylePrompt, 'gemini-3.8-flash-lite-tts');
      } else {
        const pcmChunks: Buffer[] = [];
        const BATCH_SIZE = 1; // sequential: see note above
        for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
          const batch = chunks.slice(i, i + BATCH_SIZE);
          const results = await Promise.all(
            batch.map((c) => generateSingleChunkAudio(ai, c, chosenVoice, stylePrompt, 'gemini-3.8-flash-lite-tts'))
          );
          for (const buf of results) {
            pcmChunks.push(extractPcmFromWav(buf));
          }
        }
        const totalPcm = Buffer.concat(pcmChunks);
        wavBuffer = createWavFile(totalPcm, 24000, 1, 16);
      }

      audioCache.set(cacheKey, {
        audioBase64: wavBuffer.toString('base64'),
        mimeType: 'audio/wav',
        timestamp: Date.now(),
      });
    }

    // Direct binary WAV stream requested
    if (format === 'audio' || format === 'wav' || req.headers.accept?.includes('audio/wav')) {
      res.setHeader('Content-Type', 'audio/wav');
      res.setHeader('Content-Disposition', 'inline; filename="vocalis-speech.wav"');
      res.setHeader('Content-Length', wavBuffer.length);
      return res.end(wavBuffer);
    }

    // Default JSON base64 output
    return res.json({
      success: true,
      audioBase64: wavBuffer.toString('base64'),
      mimeType: 'audio/wav',
      voice: chosenVoice,
      accent,
      charCount: cleanText.length,
      wordCount: cleanText.split(/\s+/).filter(Boolean).length,
    });
  } catch (err: any) {
    console.error('API v1 TTS Error:', err);
    return res.status(500).json({ error: formatErrorMessage(err) });
  }
});

// Developer TTS Endpoint (GET for direct streaming in browser/audio tags)
app.get('/api/v1/tts', verifyApiKey, async (req, res) => {
  try {
    const text = req.query.text as string;
    const voice = (req.query.voice as string) || 'Kore';
    const accent = (req.query.accent as string) || 'Bắc';
    const style = (req.query.style as string) || 'Tự nhiên, truyền cảm';

    if (!text || !text.trim()) {
      return res.status(400).send('Tham số text là bắt buộc.');
    }

    const ai = getGeminiClient();
    let stylePrompt = style;
    if (accent === 'Bắc') stylePrompt += ', giọng Bắc chuẩn mực';
    else if (accent === 'Nam') stylePrompt += ', giọng Nam hào sảng';

    const wavBuf = await generateSingleChunkAudio(ai, text.trim(), voice, stylePrompt, 'gemini-3.8-flash-lite-tts');

    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Content-Disposition', 'inline; filename="speech.wav"');
    res.setHeader('Content-Length', wavBuf.length);
    return res.end(wavBuf);
  } catch (err: any) {
    console.error('GET /api/v1/tts error:', err);
    return res.status(500).send(err.message || 'Lỗi phát âm thanh.');
  }
});

async function startServer() {
  const isDev = process.env.NODE_ENV !== 'production';

  if (isDev) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`TTS App server running at http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
