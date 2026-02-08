import "dotenv/config";
import express from "express";
import axios from "axios";
import pg from "pg";
import { createClient } from "@supabase/supabase-js";
import { z } from "zod";

const app = express();
app.use(express.json());

// =======================
// Config (Supabase / PostgreSQL + Storage)
// =======================
const CFG = {
    port: Number(process.env.PORT || 3000),
    db: {
        connectionString: process.env.DATABASE_URL,
        host: process.env.DB_HOST || process.env.HOST_SUPABASE,
        port: Number(process.env.DB_PORT || process.env.PORT_SUPABASE || 5432),
        database: process.env.DB_NAME || process.env.DATABASE_SUPABASE || "postgres",
        user: process.env.DB_USER || process.env.USER_SUPABASE,
        password: process.env.DB_PASSWORD || process.env.PASSWORD_SUPABASE,
        ssl: process.env.DB_SSL !== "false" ? { rejectUnauthorized: false } : false,
        max: 3,
        idleTimeoutMillis: 60000,
    },
    supabase: {
        url: process.env.SUPABASE_URL || "",
        serviceRoleKey: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY || "",
        bucket: process.env.BARCODE_CACHE_BUCKET || "barcode-cache",
    },
    provider: {
        name: "llamada_barras",
        baseUrl: process.env.LLAMADA_BARRAS_BASE_URL || "https://api.upcitemdb.com",
        lookupPath: process.env.LLAMADA_BARRAS_LOOKUP_PATH || "/prod/trial/lookup",
        apiKey: process.env.LLAMADA_BARRAS_API_KEY || "",
        apiKeyHeaderName: process.env.LLAMADA_BARRAS_APIKEY_HEADER || "key",
        timeoutMs: Number(process.env.HTTP_TIMEOUT_MS || 8000),
    },
    google: {
        apiKey: process.env.GOOGLE_API_KEY || "",
        cseId: process.env.GOOGLE_CSE_ID || "",
        timeoutMs: Number(process.env.GOOGLE_TIMEOUT_MS || 8000),
    },
    cacheTtlDays: Number(process.env.CACHE_TTL_DAYS || 90),
    imageDownloadTimeoutMs: Number(process.env.IMAGE_DOWNLOAD_TIMEOUT_MS || 10000),
};

if (!CFG.provider.apiKey) console.log("ℹ️ Modo trial UPCitemDB (sin API key). Para uso masivo configura LLAMADA_BARRAS_API_KEY.");
if (!CFG.google.apiKey || !CFG.google.cseId) {
    console.warn("⚠️ Falta GOOGLE_API_KEY y/o GOOGLE_CSE_ID (fallback de imágenes desactivado)");
}
if (!CFG.supabase.url || !CFG.supabase.serviceRoleKey) {
    console.warn("⚠️ Falta SUPABASE_URL y/o SUPABASE_SERVICE_ROLE_KEY: las imágenes se guardarán como URL externa en cache.");
}

// =======================
// DB Pool
// =======================
let pool;

/** Retorna un pool activo de PostgreSQL, reutilizable. */
function getPool() {
    if (pool) return pool;
    const config = CFG.db.connectionString
        ? { connectionString: CFG.db.connectionString, ssl: CFG.db.ssl, max: CFG.db.max, idleTimeoutMillis: CFG.db.idleTimeoutMillis }
        : { host: CFG.db.host, port: CFG.db.port, database: CFG.db.database, user: CFG.db.user, password: CFG.db.password, ssl: CFG.db.ssl, max: CFG.db.max, idleTimeoutMillis: CFG.db.idleTimeoutMillis };
    pool = new pg.Pool(config);

    // Evita que un error silencioso “mate” el pool sin enterarte
    pool.on("error", (err) => {
        console.error("PG pool error:", err);
        pool = null;
    });

    return pool;
}

// =======================
// Supabase Storage (imagen de producto en nuestro bucket)
// =======================
let supabaseClient = null;

function getSupabase() {
    if (!supabaseClient && CFG.supabase.url && CFG.supabase.serviceRoleKey) {
        supabaseClient = createClient(CFG.supabase.url, CFG.supabase.serviceRoleKey);
    }
    return supabaseClient;
}

/**
 * Descarga la imagen desde url y la sube a Supabase Storage.
 * Si falla (red, permisos, etc.), devuelve null y se usará la URL original en cache.
 */
async function downloadImageAndUploadToSupabase(imageUrl, barcode) {
    if (!imageUrl || !barcode) return null;
    const sb = getSupabase();
    if (!sb) return null;

    try {
        const res = await axios.get(imageUrl, {
            responseType: "arraybuffer",
            timeout: CFG.imageDownloadTimeoutMs,
            maxContentLength: 5 * 1024 * 1024, // 5 MB
            headers: { "User-Agent": "CodigoBarras-a-Producto/1.0" },
        });
        const buffer = Buffer.from(res.data);
        if (buffer.length < 100) return null;

        const contentType = res.headers["content-type"] || "image/jpeg";
        const ext = contentType.includes("png") ? "png" : contentType.includes("webp") ? "webp" : "jpg";
        const path = `${String(barcode).slice(0, 20)}-${Date.now()}.${ext}`;

        const { error } = await sb.storage.from(CFG.supabase.bucket).upload(path, buffer, {
            contentType,
            upsert: true,
        });
        if (error) {
            console.warn("Supabase storage upload failed:", error.message);
            return null;
        }
        const { data } = sb.storage.from(CFG.supabase.bucket).getPublicUrl(path);
        return data?.publicUrl || null;
    } catch (e) {
        console.warn("Download/upload image failed, using original URL:", e?.message || e);
        return null;
    }
}

// =======================
// Utils
// =======================
function digitsOnly(s) {
    return String(s || "").replace(/\D/g, "");
}

function isFresh(updatedAt, ttlDays) {
    if (!updatedAt) return false;
    const now = Date.now();
    const t = new Date(updatedAt).getTime();
    if (Number.isNaN(t)) return false;
    const diffDays = (now - t) / (1000 * 60 * 60 * 24);
    return diffDays <= ttlDays;
}

function computeConfidence(item) {
    const hasTitle = !!item?.title;
    const hasBrand = !!item?.brand;
    if (hasTitle && hasBrand) return 0.9;
    if (hasTitle) return 0.7;
    return 0.4;
}

function safeParseJson(s) {
    try {
        return s ? JSON.parse(s) : {};
    } catch {
        return {};
    }
}

// =======================
// Rate-limited queue (Google CSE)
// =======================
function createRateLimitedQueue({ concurrency = 1, minIntervalMs = 300, maxQueue = 2000 } = {}) {
    let running = 0;
    let lastStart = 0;
    const q = [];

    function next() {
        if (running >= concurrency) return;
        const job = q.shift();
        if (!job) return;

        const now = Date.now();
        const wait = Math.max(0, minIntervalMs - (now - lastStart));

        running++;
        setTimeout(async () => {
            lastStart = Date.now();
            try {
                const out = await job.fn();
                job.resolve(out);
            } catch (e) {
                job.reject(e);
            } finally {
                running--;
                next();
            }
        }, wait);
    }

    return {
        size: () => q.length,
        add(fn) {
            if (q.length >= maxQueue) return Promise.reject(new Error("GOOGLE_QUEUE_FULL"));
            return new Promise((resolve, reject) => {
                q.push({ fn, resolve, reject });
                next();
            });
        },
    };
}

const googleQueue = createRateLimitedQueue({
    concurrency: Number(process.env.GOOGLE_QUEUE_CONCURRENCY || 1),
    minIntervalMs: Number(process.env.GOOGLE_QUEUE_MIN_INTERVAL_MS || 350),
    maxQueue: Number(process.env.GOOGLE_QUEUE_MAX || 2000),
});

// =======================
// Google Image Search (PRO)
// =======================
const PREFERRED_DOMAINS = ["amazon.com", "alibaba.com", "walmart.com", "ebay.com"];

function buildImageQuery({ title, brand, model, upc, ean }) {
    const parts = [brand?.trim(), title?.trim(), model?.trim(), upc?.trim(), ean?.trim(), "product photo"].filter(Boolean);
    return parts.join(" ").slice(0, 120);
}

async function googleCseImageSearch({ q, site = null, num = 1 }) {
    if (!CFG.google.apiKey || !CFG.google.cseId) return null;

    const url = new URL("https://www.googleapis.com/customsearch/v1");
    url.searchParams.set("key", CFG.google.apiKey);
    url.searchParams.set("cx", CFG.google.cseId);
    url.searchParams.set("q", q);
    url.searchParams.set("searchType", "image");
    url.searchParams.set("num", String(num));
    url.searchParams.set("safe", "active");
    url.searchParams.set("imgType", "photo");
    url.searchParams.set("imgSize", "large");

    if (site) {
        url.searchParams.set("siteSearch", site);
        url.searchParams.set("siteSearchFilter", "i");
    }

    const res = await axios.get(url.toString(), { timeout: CFG.google.timeoutMs });
    const item = res?.data?.items?.[0];
    if (!item) return null;

    return {
        imageUrl: item.link,
        contextUrl: item?.image?.contextLink ?? null,
        title: item.title ?? null,
        sourceDomain: site || null,
        raw: res.data,
    };
}

async function findBestImageWithDomainPreference(input) {
    const q = buildImageQuery(input);
    const num = Number(process.env.GOOGLE_IMAGE_NUM || 1);

    for (const domain of PREFERRED_DOMAINS) {
        const r = await googleCseImageSearch({ q, site: domain, num });
        if (r?.imageUrl) {
            return { query: q, imageUrl: r.imageUrl, contextUrl: r.contextUrl, title: r.title, chosenDomain: domain, strategy: "siteSearch" };
        }
    }

    const rg = await googleCseImageSearch({ q, site: null, num });
    if (rg?.imageUrl) {
        return { query: q, imageUrl: rg.imageUrl, contextUrl: rg.contextUrl, title: rg.title, chosenDomain: null, strategy: "general" };
    }

    return null;
}

async function getGoogleImageQueued(input) {
    if (!CFG.google.apiKey || !CFG.google.cseId) return null;
    return googleQueue.add(() => findBestImageWithDomainPreference(input));
}

app.get("/health", (req, res) => {
    res.json({ ok: true, service: "codigobarras-a-producto" });
});

app.get("/health/google-queue", (req, res) => {
    res.json({
        ok: true,
        pending: googleQueue.size(),
        concurrency: Number(process.env.GOOGLE_QUEUE_CONCURRENCY || 1),
        minIntervalMs: Number(process.env.GOOGLE_QUEUE_MIN_INTERVAL_MS || 350),
    });
});

// =======================
// Normalizer
// =======================
function normalizeUPCItem(item) {
    const confidence = computeConfidence(item);
    const imageUrl = Array.isArray(item?.images) && item.images.length ? item.images[0] : null;

    return {
        name: item?.title ?? null,
        brand: item?.brand ?? null,
        model: item?.model ?? null,
        category: item?.category ?? null,
        image_url: imageUrl,
        specs: {
            description: item?.description ?? null,
            ean: item?.ean ?? null,
            upc: item?.upc ?? null,
            gtin: item?.gtin ?? null,
            asin: item?.asin ?? null,
            prices: {
                lowest_recorded_price: item?.lowest_recorded_price ?? null,
                highest_recorded_price: item?.highest_recorded_price ?? null,
            },
            offers: Array.isArray(item?.offers) ? item.offers : [],
            image_fallback: null,
        },
        confidence_score: confidence,
    };
}

// =======================
// Provider Client (UPCitemdb)
// =======================
async function lookupUPCItemDB(barcode) {
    const url = `${CFG.provider.baseUrl}${CFG.provider.lookupPath}`;
    const headers = {};
    if (CFG.provider.apiKey) {
        headers[CFG.provider.apiKeyHeaderName] = CFG.provider.apiKey;
    }

    return axios.get(url, {
        timeout: CFG.provider.timeoutMs,
        params: { upc: barcode },
        headers,
    });
}

// =======================
// Repository (PostgreSQL) — una sola tabla: barcode_cache
// =======================
async function findCachedByBarcode(barcode) {
    const p = getPool();
    const r = await p.query(
        "SELECT barcode, name, description, image_url, category, updated_at FROM barcode_cache WHERE barcode = $1 LIMIT 1",
        [barcode]
    );
    return r.rows?.[0] || null;
}

async function upsertCache(barcode, { name, description, image_url, category }) {
    const p = getPool();
    await p.query(
        `INSERT INTO barcode_cache (barcode, name, description, image_url, category, updated_at)
         VALUES ($1, $2, $3, $4, $5, NOW())
         ON CONFLICT (barcode) DO UPDATE SET
           name = EXCLUDED.name,
           description = EXCLUDED.description,
           image_url = EXCLUDED.image_url,
           category = EXCLUDED.category,
           updated_at = NOW()`,
        [barcode, name || null, description || null, image_url || null, category || "OTROS"]
    );
}

// =======================
// Validation
// =======================
const ScanSchema = z.object({
    barcode: z.string().min(3),
    symbology: z.string().optional(),
    device_id: z.string().optional(),
    user_id: z.string().optional(),
});

// =======================
// Endpoint: POST /scan
// =======================
app.post("/scan", async (req, res) => {
    const parsed = ScanSchema.safeParse(req.body);
    if (!parsed.success) {
        return res.status(400).json({ code: "INVALID_QUERY", message: parsed.error.message });
    }

    const barcode = digitsOnly(parsed.data.barcode);
    if (!barcode) return res.status(400).json({ code: "INVALID_QUERY", message: "barcode vacío" });

    try {
        // 1) Cache primero (tabla barcode_cache: código, nombre, descripción, image_url)
        const cached = await findCachedByBarcode(barcode);
        if (cached && isFresh(cached.updated_at, CFG.cacheTtlDays)) {
            return res.json({
                barcode,
                resolved_from: "DB",
                provider: "cache",
                confidence: 1,
                product: {
                    name: cached.name,
                    brand: null,
                    model: null,
                    category: cached.category || "OTROS",
                    image_url: cached.image_url,
                    specs: { description: cached.description },
                },
            });
        }

        // 2) Proveedor (UPCitemDB)
        let resp;
        try {
            resp = await lookupUPCItemDB(barcode);
        } catch (e) {
            return res.status(502).json({ code: "PROVIDER_ERR", message: "Error consultando proveedor", status: e?.response?.status ?? 500 });
        }

        const items = resp.data?.items || [];
        if (!items.length) {
            return res.status(404).json({ code: "NOT_FOUND", message: "No match was found", barcode });
        }

        const normalized = normalizeUPCItem(items[0]);
        const name = normalized.name || [normalized.brand, normalized.model].filter(Boolean).join(" ") || "Sin nombre";
        const description = (normalized.specs && typeof normalized.specs.description === "string") ? normalized.specs.description : (normalized.name || "");
        let image_url = normalized.image_url;

        // 3) Si no hay imagen del proveedor, intentar Google CSE
        if (!image_url) {
            try {
                const g = await getGoogleImageQueued({
                    title: normalized.name,
                    brand: normalized.brand,
                    model: normalized.model,
                    upc: normalized?.specs?.upc,
                    ean: normalized?.specs?.ean,
                });
                if (g?.imageUrl) image_url = g.imageUrl;
            } catch (_) {}
        }

        // 4) Imagen: intentar descargar y subir a Supabase; si falla, guardar URL original
        let finalImageUrl = image_url;
        if (image_url) {
            const supabaseUrl = await downloadImageAndUploadToSupabase(image_url, barcode);
            if (supabaseUrl) finalImageUrl = supabaseUrl;
        }

        const category = (normalized.category && String(normalized.category).trim()) ? String(normalized.category).trim() : "OTROS";

        // 5) Guardar en cache (una sola tabla)
        await upsertCache(barcode, {
            name,
            description: description || name,
            image_url: finalImageUrl,
            category,
        });

        return res.json({
            barcode,
            resolved_from: "PROVIDER",
            provider: CFG.provider.name,
            confidence: normalized.confidence_score,
            product: {
                name,
                brand: normalized.brand,
                model: normalized.model,
                category,
                image_url: finalImageUrl,
                specs: normalized.specs,
            },
        });
    } catch (err) {
        console.error(err);
        return res.status(500).json({ code: "SERVER_ERR", message: "internal server error" });
    }
});

app.listen(CFG.port, () => {
    console.log(`API running on http://localhost:${CFG.port}`);
});
