import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
app.use(express.json({ limit: "50mb" }));
const PORT = 3000;

function getAIClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// Map style presets to prompt modifiers
const STYLE_PROMPTS: Record<string, string> = {
  realistic: "hyperrealistic 8k photo, sharp focus, professional photography, natural lighting, shot on 35mm lens, depth of field",
  anime: "anime masterpiece, Makoto Shinkai style, vibrant colors, detailed lineart, dramatic lighting, high key key art",
  pixar: "3D animation character, Pixar and Disney style, soft volumetric lighting, rich textures, cute expressive features, 8k render",
  cartoon: "clean vector cartoon style, bold outlines, flat colors with smooth gradients, playful modern character design",
  fantasy: "epic high fantasy artwork, dark fantasy aesthetic, mythical atmosphere, intricate detail, magical glowing effects, Greg Rutkowski artstation",
  cyberpunk: "cyberpunk aesthetic, neon city lights, holographic reflections, moody rain-slicked surfaces, futuristic tech, synthwave color palette",
  cinematic: "cinematic still, anamorphic lens flare, dramatic contrast, movie film grain, moody atmospheric lighting, panavision",
  "3d-render": "octane 3D render, raytracing, polished smooth materials, studio studio lighting, unreal engine 5, photorealistic details",
  "oil-painting": "classic impressionist oil painting, visible thick impasto brushstrokes, rich canvas texture, rich oil colors, fine art studio",
  watercolor: "delicate watercolor painting, soft pigment bleeds, wet-on-wet technique, paper texture, subtle pastels, elegant splash art",
  sketch: "detailed pencil sketch, fine charcoal hatching, hand-drawn paper texture, artistic graphite shading, architectural draft",
  logo: "minimalist vector logo design, flat graphics, iconic emblem, isolated on clean dark background, modern branding typography, scalable graphic",
  sticker: "die-cut diecut sticker, white outline border, vibrant pop art illustration, glossy finish, cute playful vector design",
  "vector-art": "clean flat vector art, isometric perspective, crisp shapes, modern color palette, adobe illustrator graphic",
  "pixel-art": "detailed 16-bit pixel art, retro arcade game asset, crisp pixel grid, vibrant nostalgic color palette, 2D sprite"
};

// Enhances user prompt using Gemini 3.6 Flash
app.post("/api/enhance-prompt", async (req, res) => {
  try {
    const { prompt, style } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: "Prompt is required" });
    }

    const ai = getAIClient();
    if (!ai) {
      // Fallback enhancement if API key not available
      const styleSuffix = style && STYLE_PROMPTS[style] ? `, ${STYLE_PROMPTS[style]}` : "";
      return res.json({
        enhancedPrompt: `${prompt}, masterwork, cinematic composition, hyper-detailed, 8k resolution, award winning lighting${styleSuffix}`
      });
    }

    const response = await ai.models.generateContent({
      model: "gemini-3.6-flash",
      contents: `You are an expert AI image prompt engineer. Expand the following user prompt into a highly descriptive, visually rich prompt for high-end AI image generation. Keep it under 80 words, avoid buzzwords, and focus on visual subjects, lighting, perspective, texture, and mood.
      
Original Prompt: "${prompt}"
Requested Style: "${style || 'general'}"

Output ONLY the enhanced prompt, no conversational commentary.`,
    });

    const enhanced = response.text?.trim() || prompt;
    res.json({ enhancedPrompt: enhanced });
  } catch (error: any) {
    console.error("Enhance prompt error:", error);
    res.status(500).json({ error: error.message || "Failed to enhance prompt" });
  }
});

// Primary Image Generation Endpoint
app.post("/api/generate-image", async (req, res) => {
  try {
    const {
      prompt,
      negativePrompt,
      aspectRatio = "1:1",
      imageSize = "1K", // 1K, 2K, 4K
      model = "gemini-3.1-flash-image", // gemini-3-pro-image, gemini-3.1-flash-image, gemini-3.1-flash-lite-image
      stylePreset,
      numberOfImages = 1,
      seed,
      creativityLevel = 0.7,
    } = req.body;

    if (!prompt) {
      return res.status(400).json({ error: "Prompt is required" });
    }

    const ai = getAIClient();
    if (!ai) {
      return res.status(400).json({
        error: "GEMINI_API_KEY environment variable is missing. Please set your API key in Settings > Secrets.",
      });
    }

    // Build complete prompt with style preset and negative guidance
    let fullPrompt = prompt;
    if (stylePreset && STYLE_PROMPTS[stylePreset]) {
      fullPrompt += `, ${STYLE_PROMPTS[stylePreset]}`;
    }
    if (negativePrompt) {
      fullPrompt += `. Avoid: ${negativePrompt}`;
    }

    // Determine target model name according to guidelines
    let selectedModel = model;
    if (model === "gemini-3-pro-image-preview" || model === "gemini-3-pro-image") {
      selectedModel = "gemini-3-pro-image";
    } else if (model === "gemini-3.1-flash-image" || imageSize === "2K" || imageSize === "4K") {
      selectedModel = "gemini-3.1-flash-image";
    } else if (model === "gemini-3.1-flash-lite-image") {
      selectedModel = "gemini-3.1-flash-lite-image";
    } else {
      selectedModel = "gemini-3.1-flash-image";
    }

    // Generate count images in parallel if requested
    const count = Math.min(Math.max(1, Number(numberOfImages) || 1), 4);
    const generateOne = async (index: number) => {
      const currentSeed = seed !== undefined ? Number(seed) + index : undefined;
      const response = await ai.models.generateContent({
        model: selectedModel,
        contents: {
          parts: [{ text: fullPrompt }],
        },
        config: {
          seed: currentSeed,
          imageConfig: {
            aspectRatio: aspectRatio as any,
            imageSize: (selectedModel === "gemini-3.1-flash-lite-image" ? undefined : imageSize) as any,
          },
        },
      });

      let imageUrl = "";
      let textOutput = "";

      const parts = response.candidates?.[0]?.content?.parts || [];
      for (const part of parts) {
        if (part.inlineData) {
          const base64Data = part.inlineData.data;
          const mimeType = part.inlineData.mimeType || "image/png";
          imageUrl = `data:${mimeType};base64,${base64Data}`;
        } else if (part.text) {
          textOutput += part.text;
        }
      }

      if (!imageUrl) {
        throw new Error("No image data returned from Gemini API");
      }

      return {
        id: `gen-${Date.now()}-${index}-${Math.random().toString(36).substring(2, 7)}`,
        imageUrl,
        prompt: fullPrompt,
        originalPrompt: prompt,
        negativePrompt,
        model: selectedModel,
        aspectRatio,
        imageSize,
        stylePreset,
        createdAt: new Date().toISOString(),
        details: textOutput || undefined,
      };
    };

    const results = await Promise.all(
      Array.from({ length: count }).map((_, i) => generateOne(i))
    );

    res.json({ images: results });
  } catch (error: any) {
    console.error("Image generation error:", error);
    res.status(500).json({
      error: error.message || "An error occurred while generating images.",
    });
  }
});

// Image Editing / Variation / Upscale / Background Removal Endpoint
app.post("/api/edit-image", async (req, res) => {
  try {
    const {
      imageBase64,
      mimeType = "image/png",
      prompt,
      actionType = "edit", // 'edit', 'upscale', 'remove-bg', 'variation'
      aspectRatio = "1:1",
      imageSize = "2K",
    } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: "Base64 image is required for image editing" });
    }

    const ai = getAIClient();
    if (!ai) {
      return res.status(400).json({
        error: "GEMINI_API_KEY environment variable is missing.",
      });
    }

    let editPrompt = prompt || "Enhance this image";
    if (actionType === "upscale") {
      editPrompt = `Re-render this image in ultra-high resolution ${imageSize} crisp details, keeping exact subjects, colors and composition intact but sharply upscaled. ${prompt || ''}`;
    } else if (actionType === "remove-bg") {
      editPrompt = `Isolate the main subject of this image onto a solid pure black studio background or transparent background, removing any busy environment. ${prompt || ''}`;
    } else if (actionType === "variation") {
      editPrompt = `Create a creative artistic variation of this image with the same core subjects, enhanced dramatic lighting and stylish aesthetic. ${prompt || ''}`;
    }

    // Clean base64 data string if it includes data URL prefix
    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-image",
      contents: {
        parts: [
          {
            inlineData: {
              data: cleanBase64,
              mimeType,
            },
          },
          {
            text: editPrompt,
          },
        ],
      },
      config: {
        imageConfig: {
          aspectRatio: aspectRatio as any,
          imageSize: imageSize as any,
        },
      },
    });

    let imageUrl = "";
    const parts = response.candidates?.[0]?.content?.parts || [];
    for (const part of parts) {
      if (part.inlineData) {
        const base64Data = part.inlineData.data;
        const mime = part.inlineData.mimeType || "image/png";
        imageUrl = `data:${mime};base64,${base64Data}`;
      }
    }

    if (!imageUrl) {
      throw new Error("Failed to produce edited image");
    }

    res.json({
      imageUrl,
      actionType,
      prompt: editPrompt,
      id: `edit-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
    });
  } catch (error: any) {
    console.error("Image edit error:", error);
    res.status(500).json({ error: error.message || "Failed to process image" });
  }
});

async function startServer() {
  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`AI Studio Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
