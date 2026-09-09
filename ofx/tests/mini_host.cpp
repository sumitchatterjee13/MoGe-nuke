// Minimal OFX host for testing MoGe3.ofx without Nuke.
//
// Implements just enough of the property / parameter / image-effect / message
// suites to load the plugin, describe it, create an instance, set parameters
// and render one frame from a raw float RGB file to a raw float RGBA file.
//
//   mini_host <plugin.ofx> <in.rgb> <W> <H> <out.rgba> [name=value ...]
//
// in.rgb : W*H*3 float32, top row first (what the Python driver writes)
// out.rgba: W*H*4 float32, top row first
// name=value sets a parameter before rendering (choice/int/bool: integer
// index, double: number, string: text). "output" may be set twice by passing
// out2=<path> to render a second time and check the cache path.
//
// Not a general host: single clip, single frame, float RGBA only.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <algorithm>

// ---------------------------------------------------------------- properties

struct PropValue {
    std::vector<std::string> strings;
    std::vector<double> doubles;
    std::vector<int> ints;
    std::vector<void *> pointers;
};

struct PropSet {
    std::map<std::string, PropValue> props;
    std::vector<std::shared_ptr<std::string>> stringPool;  // stable char* for getString
};

static PropSet *ps(OfxPropertySetHandle h) { return (PropSet *)h; }

template <class V>
static void setAt(std::vector<V> &v, int index, const V &value) {
    if ((int)v.size() <= index) v.resize(index + 1);
    v[index] = value;
}

static OfxStatus propSetPointer(OfxPropertySetHandle p, const char *n, int i, void *v) { setAt(ps(p)->props[n].pointers, i, v); return kOfxStatOK; }
static OfxStatus propSetString(OfxPropertySetHandle p, const char *n, int i, const char *v) { setAt(ps(p)->props[n].strings, i, std::string(v ? v : "")); return kOfxStatOK; }
static OfxStatus propSetDouble(OfxPropertySetHandle p, const char *n, int i, double v) { setAt(ps(p)->props[n].doubles, i, v); return kOfxStatOK; }
static OfxStatus propSetInt(OfxPropertySetHandle p, const char *n, int i, int v) { setAt(ps(p)->props[n].ints, i, v); return kOfxStatOK; }
static OfxStatus propSetPointerN(OfxPropertySetHandle p, const char *n, int c, void *const *v) { for (int i = 0; i < c; ++i) propSetPointer(p, n, i, v[i]); return kOfxStatOK; }
static OfxStatus propSetStringN(OfxPropertySetHandle p, const char *n, int c, const char *const *v) { for (int i = 0; i < c; ++i) propSetString(p, n, i, v[i]); return kOfxStatOK; }
static OfxStatus propSetDoubleN(OfxPropertySetHandle p, const char *n, int c, const double *v) { for (int i = 0; i < c; ++i) propSetDouble(p, n, i, v[i]); return kOfxStatOK; }
static OfxStatus propSetIntN(OfxPropertySetHandle p, const char *n, int c, const int *v) { for (int i = 0; i < c; ++i) propSetInt(p, n, i, v[i]); return kOfxStatOK; }

static OfxStatus propGetPointer(OfxPropertySetHandle p, const char *n, int i, void **v) {
    auto it = ps(p)->props.find(n);
    if (it == ps(p)->props.end() || (int)it->second.pointers.size() <= i) { *v = nullptr; return kOfxStatErrUnknown; }
    *v = it->second.pointers[i];
    return kOfxStatOK;
}
static OfxStatus propGetString(OfxPropertySetHandle p, const char *n, int i, char **v) {
    auto it = ps(p)->props.find(n);
    if (it == ps(p)->props.end() || (int)it->second.strings.size() <= i) { *v = nullptr; return kOfxStatErrUnknown; }
    ps(p)->stringPool.push_back(std::make_shared<std::string>(it->second.strings[i]));
    *v = (char *)ps(p)->stringPool.back()->c_str();
    return kOfxStatOK;
}
static OfxStatus propGetDouble(OfxPropertySetHandle p, const char *n, int i, double *v) {
    auto it = ps(p)->props.find(n);
    if (it == ps(p)->props.end() || (int)it->second.doubles.size() <= i) { *v = 0; return kOfxStatErrUnknown; }
    *v = it->second.doubles[i];
    return kOfxStatOK;
}
static OfxStatus propGetInt(OfxPropertySetHandle p, const char *n, int i, int *v) {
    auto it = ps(p)->props.find(n);
    if (it == ps(p)->props.end() || (int)it->second.ints.size() <= i) { *v = 0; return kOfxStatErrUnknown; }
    *v = it->second.ints[i];
    return kOfxStatOK;
}
static OfxStatus propGetPointerN(OfxPropertySetHandle p, const char *n, int c, void **v) { for (int i = 0; i < c; ++i) propGetPointer(p, n, i, v + i); return kOfxStatOK; }
static OfxStatus propGetStringN(OfxPropertySetHandle p, const char *n, int c, char **v) { for (int i = 0; i < c; ++i) propGetString(p, n, i, v + i); return kOfxStatOK; }
static OfxStatus propGetDoubleN(OfxPropertySetHandle p, const char *n, int c, double *v) { for (int i = 0; i < c; ++i) propGetDouble(p, n, i, v + i); return kOfxStatOK; }
static OfxStatus propGetIntN(OfxPropertySetHandle p, const char *n, int c, int *v) { for (int i = 0; i < c; ++i) propGetInt(p, n, i, v + i); return kOfxStatOK; }
static OfxStatus propReset(OfxPropertySetHandle p, const char *n) { ps(p)->props.erase(n); return kOfxStatOK; }
static OfxStatus propGetDimension(OfxPropertySetHandle p, const char *n, int *c) {
    auto it = ps(p)->props.find(n);
    if (it == ps(p)->props.end()) { *c = 0; return kOfxStatOK; }
    auto &v = it->second;
    size_t m = v.strings.size();
    m = (std::max)(m, v.doubles.size());
    m = (std::max)(m, v.ints.size());
    m = (std::max)(m, v.pointers.size());
    *c = (int)m;
    return kOfxStatOK;
}

static OfxPropertySuiteV1 gPropSuite = {
    propSetPointer, propSetString, propSetDouble, propSetInt,
    propSetPointerN, propSetStringN, propSetDoubleN, propSetIntN,
    propGetPointer, propGetString, propGetDouble, propGetInt,
    propGetPointerN, propGetStringN, propGetDoubleN, propGetIntN,
    propReset, propGetDimension,
};

// ---------------------------------------------------------------- effect

struct Param {
    std::string type;
    PropSet props;         // descriptor props (defaults, options ...)
    int i = 0;
    double d = 0;
    std::string s;
};

struct Image {
    PropSet props;
    std::vector<float> data;
};

struct Clip {
    PropSet props;
    std::unique_ptr<Image> image;
};

struct Effect {  // used for both the descriptor and the instance
    PropSet props;
    PropSet paramSetProps;
    std::map<std::string, std::unique_ptr<Param>> params;
    std::vector<std::string> paramOrder;
    std::map<std::string, std::unique_ptr<Clip>> clips;
};

static Effect *ef(OfxImageEffectHandle h) { return (Effect *)h; }

static OfxStatus getPropertySet(OfxImageEffectHandle e, OfxPropertySetHandle *p) { *p = (OfxPropertySetHandle)&ef(e)->props; return kOfxStatOK; }
static OfxStatus getParamSet(OfxImageEffectHandle e, OfxParamSetHandle *p) { *p = (OfxParamSetHandle)e; return kOfxStatOK; }
static OfxStatus clipDefine(OfxImageEffectHandle e, const char *name, OfxPropertySetHandle *p) {
    auto &c = ef(e)->clips[name];
    if (!c) c.reset(new Clip());
    if (p) *p = (OfxPropertySetHandle)&c->props;
    return kOfxStatOK;
}
static OfxStatus clipGetHandle(OfxImageEffectHandle e, const char *name, OfxImageClipHandle *h, OfxPropertySetHandle *p) {
    auto it = ef(e)->clips.find(name);
    if (it == ef(e)->clips.end()) return kOfxStatErrBadHandle;
    *h = (OfxImageClipHandle)it->second.get();
    if (p) *p = (OfxPropertySetHandle)&it->second->props;
    return kOfxStatOK;
}
static OfxStatus clipGetPropertySet(OfxImageClipHandle c, OfxPropertySetHandle *p) { *p = (OfxPropertySetHandle)&((Clip *)c)->props; return kOfxStatOK; }
static OfxStatus clipGetImage(OfxImageClipHandle c, OfxTime, const OfxRectD *, OfxPropertySetHandle *img) {
    Clip *clip = (Clip *)c;
    if (!clip->image) return kOfxStatFailed;
    *img = (OfxPropertySetHandle)&clip->image->props;
    return kOfxStatOK;
}
static OfxStatus clipReleaseImage(OfxPropertySetHandle) { return kOfxStatOK; }
static OfxStatus clipGetRegionOfDefinition(OfxImageClipHandle c, OfxTime, OfxRectD *rod) {
    Clip *clip = (Clip *)c;
    if (!clip->image) return kOfxStatFailed;
    int b[4];
    propGetIntN((OfxPropertySetHandle)&clip->image->props, kOfxImagePropBounds, 4, b);
    rod->x1 = b[0]; rod->y1 = b[1]; rod->x2 = b[2]; rod->y2 = b[3];
    return kOfxStatOK;
}
static int effectAbort(OfxImageEffectHandle) { return 0; }
static OfxStatus imageMemoryAlloc(OfxImageEffectHandle, size_t n, OfxImageMemoryHandle *h) { *h = (OfxImageMemoryHandle)malloc(n); return kOfxStatOK; }
static OfxStatus imageMemoryFree(OfxImageMemoryHandle h) { free(h); return kOfxStatOK; }
static OfxStatus imageMemoryLock(OfxImageMemoryHandle h, void **p) { *p = h; return kOfxStatOK; }
static OfxStatus imageMemoryUnlock(OfxImageMemoryHandle) { return kOfxStatOK; }

static OfxImageEffectSuiteV1 gEffectSuite = {
    getPropertySet, getParamSet, clipDefine, clipGetHandle, clipGetPropertySet, clipGetImage,
    clipReleaseImage, clipGetRegionOfDefinition, effectAbort,
    imageMemoryAlloc, imageMemoryFree, imageMemoryLock, imageMemoryUnlock,
};

// ---------------------------------------------------------------- params

static OfxStatus paramDefine(OfxParamSetHandle set, const char *type, const char *name, OfxPropertySetHandle *p) {
    Effect *e = (Effect *)set;
    auto &prm = e->params[name];
    prm.reset(new Param());
    prm->type = type;
    e->paramOrder.push_back(name);
    if (p) *p = (OfxPropertySetHandle)&prm->props;
    return kOfxStatOK;
}
static OfxStatus paramGetHandle(OfxParamSetHandle set, const char *name, OfxParamHandle *h, OfxPropertySetHandle *p) {
    Effect *e = (Effect *)set;
    auto it = e->params.find(name);
    if (it == e->params.end()) return kOfxStatErrUnknown;
    *h = (OfxParamHandle)it->second.get();
    if (p) *p = (OfxPropertySetHandle)&it->second->props;
    return kOfxStatOK;
}
static OfxStatus paramSetGetPropertySet(OfxParamSetHandle set, OfxPropertySetHandle *p) { *p = (OfxPropertySetHandle)&((Effect *)set)->paramSetProps; return kOfxStatOK; }
static OfxStatus paramGetPropertySet(OfxParamHandle h, OfxPropertySetHandle *p) { *p = (OfxPropertySetHandle)&((Param *)h)->props; return kOfxStatOK; }

static OfxStatus fillValue(Param *prm, va_list ap) {
    if (prm->type == kOfxParamTypeDouble) *va_arg(ap, double *) = prm->d;
    else if (prm->type == kOfxParamTypeString) *va_arg(ap, char **) = (char *)prm->s.c_str();
    else *va_arg(ap, int *) = prm->i;  // integer, boolean, choice
    return kOfxStatOK;
}
static OfxStatus paramGetValue(OfxParamHandle h, ...) { va_list ap; va_start(ap, h); OfxStatus s = fillValue((Param *)h, ap); va_end(ap); return s; }
static OfxStatus paramGetValueAtTime(OfxParamHandle h, OfxTime t, ...) { va_list ap; va_start(ap, t); OfxStatus s = fillValue((Param *)h, ap); va_end(ap); return s; }
static OfxStatus paramUnsupported(OfxParamHandle, ...) { return kOfxStatErrUnsupported; }
static OfxStatus paramUnsupportedT(OfxParamHandle, OfxTime, ...) { return kOfxStatErrUnsupported; }
static OfxStatus paramUnsupportedTT(OfxParamHandle, OfxTime, OfxTime, ...) { return kOfxStatErrUnsupported; }
static OfxStatus paramGetNumKeys(OfxParamHandle, unsigned int *n) { *n = 0; return kOfxStatOK; }
static OfxStatus paramGetKeyTime(OfxParamHandle, unsigned int, OfxTime *) { return kOfxStatErrUnsupported; }
static OfxStatus paramGetKeyIndex(OfxParamHandle, OfxTime, int, int *) { return kOfxStatErrUnsupported; }
static OfxStatus paramDeleteKey(OfxParamHandle, OfxTime) { return kOfxStatOK; }
static OfxStatus paramDeleteAllKeys(OfxParamHandle) { return kOfxStatOK; }
static OfxStatus paramCopy(OfxParamHandle, OfxParamHandle, OfxTime, const OfxRangeD *) { return kOfxStatErrUnsupported; }
static OfxStatus paramEditBegin(OfxParamSetHandle, const char *) { return kOfxStatOK; }
static OfxStatus paramEditEnd(OfxParamSetHandle) { return kOfxStatOK; }

static OfxParameterSuiteV1 gParamSuite = {
    paramDefine, paramGetHandle, paramSetGetPropertySet, paramGetPropertySet,
    paramGetValue, paramGetValueAtTime,
    paramUnsupportedT /*derivative*/, paramUnsupportedTT /*integral*/,
    paramUnsupported /*setValue*/, paramUnsupportedT /*setValueAtTime*/,
    paramGetNumKeys, paramGetKeyTime, paramGetKeyIndex, paramDeleteKey, paramDeleteAllKeys,
    paramCopy, paramEditBegin, paramEditEnd,
};

// ---------------------------------------------------------------- message

static OfxStatus hostMessage(void *, const char *type, const char *, const char *fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[host] plugin message (%s): %s\n", type, buf);
    return kOfxStatOK;
}
static OfxMessageSuiteV1 gMessageSuite = {hostMessage};

// ---------------------------------------------------------------- host

static PropSet gHostProps;
static const void *fetchSuite(OfxPropertySetHandle, const char *name, int version) {
    if (strcmp(name, kOfxPropertySuite) == 0 && version == 1) return &gPropSuite;
    if (strcmp(name, kOfxImageEffectSuite) == 0 && version == 1) return &gEffectSuite;
    if (strcmp(name, kOfxParameterSuite) == 0 && version == 1) return &gParamSuite;
    if (strcmp(name, kOfxMessageSuite) == 0 && version == 1) return &gMessageSuite;
    return nullptr;  // progress etc: the plugin must cope without
}
static OfxHost gHost = {(OfxPropertySetHandle)&gHostProps, fetchSuite};

static Image *makeImage(int w, int h, const char *comps, int ncomp) {
    Image *img = new Image();
    img->data.assign((size_t)w * h * ncomp, 0.f);
    OfxPropertySetHandle p = (OfxPropertySetHandle)&img->props;
    int bounds[4] = {0, 0, w, h};
    propSetIntN(p, kOfxImagePropBounds, 4, bounds);
    propSetIntN(p, kOfxImagePropRegionOfDefinition, 4, bounds);
    propSetInt(p, kOfxImagePropRowBytes, 0, w * ncomp * (int)sizeof(float));
    propSetPointer(p, kOfxImagePropData, 0, img->data.data());
    propSetString(p, kOfxImageEffectPropComponents, 0, comps);
    propSetString(p, kOfxImageEffectPropPixelDepth, 0, kOfxBitDepthFloat);
    propSetString(p, kOfxImageEffectPropPreMultiplication, 0, kOfxImageUnPreMultiplied);
    propSetDouble(p, kOfxImagePropPixelAspectRatio, 0, 1.0);
    propSetString(p, kOfxImagePropField, 0, kOfxImageFieldNone);
    return img;
}

static bool readRaw(const char *path, std::vector<float> &out, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    out.resize(n);
    size_t got = fread(out.data(), sizeof(float), n, f);
    fclose(f);
    return got == n;
}

static bool writeRaw(const char *path, const std::vector<float> &v) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    return true;
}

static OfxStatus render(OfxPlugin *plugin, Effect *inst, int w, int h) {
    PropSet inArgs;
    OfxPropertySetHandle ia = (OfxPropertySetHandle)&inArgs;
    propSetDouble(ia, kOfxPropTime, 0, 1.0);
    int rw[4] = {0, 0, w, h};
    propSetIntN(ia, kOfxImageEffectPropRenderWindow, 4, rw);
    double rs[2] = {1.0, 1.0};
    propSetDoubleN(ia, kOfxImageEffectPropRenderScale, 2, rs);
    propSetString(ia, kOfxImageEffectPropFieldToRender, 0, kOfxImageFieldNone);
    propSetInt(ia, kOfxPropIsInteractive, 0, 0);
    return plugin->mainEntry(kOfxImageEffectActionRender, inst, ia, nullptr);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: mini_host plugin.ofx in.rgb W H out.rgba [name=value ...] [out2=path]\n");
        return 2;
    }
    const char *pluginPath = argv[1];
    const char *inPath = argv[2];
    int W = atoi(argv[3]), H = atoi(argv[4]);
    const char *outPath = argv[5];

#ifdef _WIN32
    HMODULE lib = LoadLibraryA(pluginPath);
    if (!lib) { fprintf(stderr, "LoadLibrary failed (%lu)\n", GetLastError()); return 1; }
    auto getN = (int (*)(void))GetProcAddress(lib, "OfxGetNumberOfPlugins");
    auto getP = (OfxPlugin * (*)(int))GetProcAddress(lib, "OfxGetPlugin");
#else
    void *lib = dlopen(pluginPath, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    auto getN = (int (*)(void))dlsym(lib, "OfxGetNumberOfPlugins");
    auto getP = (OfxPlugin * (*)(int))dlsym(lib, "OfxGetPlugin");
#endif
    if (!getN || !getP || getN() < 1) { fprintf(stderr, "no OFX entry points\n"); return 1; }
    OfxPlugin *plugin = getP(0);
    printf("[host] plugin %s v%d.%d api %s/%d\n", plugin->pluginIdentifier, plugin->pluginVersionMajor,
           plugin->pluginVersionMinor, plugin->pluginApi, plugin->apiVersion);

    propSetString((OfxPropertySetHandle)&gHostProps, kOfxPropName, 0, "com.sumit.moge3.minihost");
    propSetString((OfxPropertySetHandle)&gHostProps, kOfxPropLabel, 0, "MoGe3 mini host");
    plugin->setHost(&gHost);
    if (plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) != kOfxStatOK) { fprintf(stderr, "load failed\n"); return 1; }

    Effect desc;
    if (plugin->mainEntry(kOfxActionDescribe, &desc, nullptr, nullptr) != kOfxStatOK) { fprintf(stderr, "describe failed\n"); return 1; }
    char *label = nullptr;
    propGetString((OfxPropertySetHandle)&desc.props, kOfxPropLabel, 0, &label);
    int tiles = -1;
    propGetInt((OfxPropertySetHandle)&desc.props, kOfxImageEffectPropSupportsTiles, 0, &tiles);
    printf("[host] label '%s' supportsTiles=%d\n", label ? label : "?", tiles);

    Effect ctx;
    PropSet ctxArgs;
    propSetString((OfxPropertySetHandle)&ctxArgs, kOfxImageEffectPropContext, 0, kOfxImageEffectContextFilter);
    if (plugin->mainEntry(kOfxImageEffectActionDescribeInContext, &ctx, (OfxPropertySetHandle)&ctxArgs, nullptr) != kOfxStatOK) {
        fprintf(stderr, "describeInContext failed\n");
        return 1;
    }
    printf("[host] %zu params, %zu clips\n", ctx.params.size(), ctx.clips.size());

    // instance: params start at their descriptor defaults
    Effect inst;
    for (auto &name : ctx.paramOrder) {
        Param *d = ctx.params[name].get();
        std::unique_ptr<Param> p(new Param());
        p->type = d->type;
        p->props = d->props;
        OfxPropertySetHandle dp = (OfxPropertySetHandle)&d->props;
        if (p->type == kOfxParamTypeDouble) propGetDouble(dp, kOfxParamPropDefault, 0, &p->d);
        else if (p->type == kOfxParamTypeString) { char *s = nullptr; propGetString(dp, kOfxParamPropDefault, 0, &s); p->s = s ? s : ""; }
        else propGetInt(dp, kOfxParamPropDefault, 0, &p->i);
        inst.params[name] = std::move(p);
        inst.paramOrder.push_back(name);
    }
    for (auto &kv : ctx.clips) inst.clips[kv.first].reset(new Clip());
    std::string out2;
    for (int i = 6; i < argc; ++i) {
        std::string a = argv[i];
        size_t eq = a.find('=');
        if (eq == std::string::npos) continue;
        std::string k = a.substr(0, eq), v = a.substr(eq + 1);
        if (k == "out2") { out2 = v; continue; }
        auto it = inst.params.find(k);
        if (it == inst.params.end()) { fprintf(stderr, "[host] no such param %s\n", k.c_str()); return 2; }
        if (it->second->type == kOfxParamTypeDouble) it->second->d = atof(v.c_str());
        else if (it->second->type == kOfxParamTypeString) it->second->s = v;
        else it->second->i = atoi(v.c_str());
        printf("[host] set %s = %s\n", k.c_str(), v.c_str());
    }
    for (auto &name : inst.paramOrder) {
        Param *p = inst.params[name].get();
        if (p->type == kOfxParamTypeDouble) printf("[host]   %-16s %g\n", name.c_str(), p->d);
        else if (p->type == kOfxParamTypeString) printf("[host]   %-16s '%s'\n", name.c_str(), p->s.c_str());
        else printf("[host]   %-16s %d\n", name.c_str(), p->i);
    }

    if (plugin->mainEntry(kOfxActionCreateInstance, &inst, nullptr, nullptr) != kOfxStatOK) { fprintf(stderr, "createInstance failed\n"); return 1; }

    // clip preferences (the plugin asks for RGBA float out)
    PropSet prefs;
    plugin->mainEntry(kOfxImageEffectActionGetClipPreferences, &inst, nullptr, (OfxPropertySetHandle)&prefs);

    // source image: file is top-first, OFX is bottom-up
    std::vector<float> rgb;
    if (!readRaw(inPath, rgb, (size_t)W * H * 3)) { fprintf(stderr, "cannot read %s\n", inPath); return 1; }
    Image *src = makeImage(W, H, kOfxImageComponentRGBA, 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float *s = &rgb[((size_t)y * W + x) * 3];
            float *d = &src->data[((size_t)(H - 1 - y) * W + x) * 4];
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 1.f;
        }
    inst.clips[kOfxImageEffectSimpleSourceClipName]->image.reset(src);
    Image *dst = makeImage(W, H, kOfxImageComponentRGBA, 4);
    inst.clips[kOfxImageEffectOutputClipName]->image.reset(dst);

    // regions of interest: the plugin should ask for the whole source
    {
        PropSet ia, oa;
        propSetDouble((OfxPropertySetHandle)&ia, kOfxPropTime, 0, 1.0);
        double roi[4] = {10, 10, 20, 20};
        propSetDoubleN((OfxPropertySetHandle)&ia, kOfxImageEffectPropRegionOfInterest, 4, roi);
        double rs[2] = {1, 1};
        propSetDoubleN((OfxPropertySetHandle)&ia, kOfxImageEffectPropRenderScale, 2, rs);
        plugin->mainEntry(kOfxImageEffectActionGetRegionsOfInterest, &inst, (OfxPropertySetHandle)&ia, (OfxPropertySetHandle)&oa);
        double got[4] = {0, 0, 0, 0};
        propGetDoubleN((OfxPropertySetHandle)&oa, "OfxImageClipPropRoI_Source", 4, got);
        printf("[host] RoI for a 10x10 request -> source %g,%g %g,%g (expect full frame)\n", got[0], got[1], got[2], got[3]);
    }

    auto dump = [&](const char *path) {
        std::vector<float> out((size_t)W * H * 4);
        for (int y = 0; y < H; ++y)
            memcpy(&out[(size_t)y * W * 4], &dst->data[(size_t)(H - 1 - y) * W * 4], (size_t)W * 4 * sizeof(float));
        return writeRaw(path, out);
    };

    OfxStatus st = render(plugin, &inst, W, H);
    printf("[host] render 1 -> status %d\n", st);
    if (st != kOfxStatOK) return 1;
    if (!dump(outPath)) { fprintf(stderr, "cannot write %s\n", outPath); return 1; }

    if (!out2.empty()) {
        inst.params["output"]->i = 1 - inst.params["output"]->i;  // flip depth <-> normals
        st = render(plugin, &inst, W, H);
        printf("[host] render 2 (output flipped, should be a cache hit) -> status %d\n", st);
        if (st != kOfxStatOK) return 1;
        if (!dump(out2.c_str())) return 1;
    }

    plugin->mainEntry(kOfxActionDestroyInstance, &inst, nullptr, nullptr);
    plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr);
    printf("[host] OK\n");
    return 0;
}
