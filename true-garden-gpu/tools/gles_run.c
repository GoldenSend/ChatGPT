// Minimal headless GLES 3.2 compute runner (EGL surfaceless / device platform).
// Used offline with Mesa LLVMPipe; the same binary is meant to run on the PS4
// against radeonsi only after an owner-approved hardware gate.
//
// usage: gles_run --src FILE [--src FILE ...] --groups X
//                 [--in BINDING:PATH ...] [--out BINDING:PATH:BYTES ...]
//                 [--int NAME=VALUE ...]
// Sources are concatenated after "#version 320 es\n". Input files are
// uploaded verbatim to SSBO BINDING; outputs are read back after the
// dispatch. Prints GL_RENDERER so logs say which driver produced a result.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXB 16

static char * slurp(const char * path, long * n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * b = malloc(*n + 1);
    if (fread(b, 1, *n, f) != (size_t) *n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    b[*n] = 0;
    fclose(f);
    return b;
}

int main(int argc, char ** argv) {
    const char * srcs[32]; int nsrc = 0;
    int groups = 1;
    int in_b[MAXB], out_b[MAXB], nin = 0, nout = 0;
    const char * in_p[MAXB]; const char * out_p[MAXB]; long out_n[MAXB];
    const char * int_name[MAXB]; int int_val[MAXB]; int nint = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--src")) srcs[nsrc++] = argv[++i];
        else if (!strcmp(argv[i], "--groups")) groups = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--in")) {
            char * s = argv[++i]; char * c = strchr(s, ':'); *c = 0;
            in_b[nin] = atoi(s); in_p[nin++] = c + 1;
        } else if (!strcmp(argv[i], "--out")) {
            char * s = argv[++i]; char * c1 = strchr(s, ':'); *c1 = 0; char * c2 = strrchr(c1 + 1, ':'); *c2 = 0;
            out_b[nout] = atoi(s); out_p[nout] = c1 + 1; out_n[nout++] = atol(c2 + 1);
        } else if (!strcmp(argv[i], "--int")) {
            char * s = argv[++i]; char * c = strchr(s, '='); *c = 0;
            int_name[nint] = s; int_val[nint++] = atoi(c + 1);
        } else { fprintf(stderr, "bad arg %s\n", argv[i]); return 1; }
    }

    PFNEGLGETPLATFORMDISPLAYEXTPROC getdpy = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = getdpy ? getdpy(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL) : EGL_NO_DISPLAY;
    if (dpy == EGL_NO_DISPLAY) dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint maj, min;
    if (!eglInitialize(dpy, &maj, &min)) { fprintf(stderr, "eglInitialize failed\n"); return 3; }
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint cfg_attr[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
    EGLConfig cfg; EGLint ncfg = 0;
    eglChooseConfig(dpy, cfg_attr, &cfg, 1, &ncfg);
    const EGLint ctx_attr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, ncfg ? cfg : EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr);
    if (ctx == EGL_NO_CONTEXT) { fprintf(stderr, "eglCreateContext failed 0x%x\n", eglGetError()); return 3; }
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
    fprintf(stderr, "GL_RENDERER: %s | GL_VERSION: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));

    // build source
    size_t cap = 1 << 20, len = 0;
    char * src = malloc(cap);
    len += sprintf(src, "#version 320 es\n");
    for (int i = 0; i < nsrc; i++) {
        long n; char * b = slurp(srcs[i], &n);
        memcpy(src + len, b, n); len += n; src[len++] = '\n'; free(b);
    }
    src[len] = 0;
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    const char * csrc = src;
    glShaderSource(sh, 1, &csrc, NULL);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[8192]; glGetShaderInfoLog(sh, sizeof log, NULL, log); fprintf(stderr, "compile:\n%s\n", log); return 4; }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh); glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { char log[8192]; glGetProgramInfoLog(prog, sizeof log, NULL, log); fprintf(stderr, "link:\n%s\n", log); return 4; }
    glUseProgram(prog);
    for (int i = 0; i < nint; i++) glUniform1i(glGetUniformLocation(prog, int_name[i]), int_val[i]);

    GLuint bufs[2 * MAXB]; int nb = 0;
    for (int i = 0; i < nin; i++) {
        long n; char * b = slurp(in_p[i], &n);
        glGenBuffers(1, &bufs[nb]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[nb]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, n, b, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, in_b[i], bufs[nb++]);
        free(b);
    }
    for (int i = 0; i < nout; i++) {
        glGenBuffers(1, &bufs[nb]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[nb]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, out_n[i], NULL, GL_DYNAMIC_READ);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, out_b[i], bufs[nb]);
        out_b[i] = nb++;  // remember buffer index
    }
    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glFinish();
    for (int i = 0; i < nout; i++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[out_b[i]]);
        void * p = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, out_n[i], GL_MAP_READ_BIT);
        if (!p) { fprintf(stderr, "map failed 0x%x\n", glGetError()); return 5; }
        FILE * f = fopen(out_p[i], "wb"); fwrite(p, 1, out_n[i], f); fclose(f);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) { fprintf(stderr, "GL error 0x%x\n", err); return 6; }
    return 0;
}
