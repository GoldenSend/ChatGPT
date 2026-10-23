#!/usr/bin/env python3
"""ctypes twin of tools/gles_run.c with the same command line, for machines
without a C compiler or EGL/GLES headers (the PS4). Needs only python3 and
the runtime libEGL.so.1 / libGLESv2.so.2 (plus libgbm.so.1 for GBM).

usage: gles_run.py --src FILE [--src FILE ...] --groups X
                   [--in BINDING:PATH ...] [--out BINDING:PATH:BYTES ...]
                   [--int NAME=VALUE ...] [--compile-only]
Display: EGL surfaceless by default. With GLES_DRM_DEVICE=/dev/dri/renderD128
it opens that node through GBM and uses EGL_PLATFORM_GBM, the way the GPU-021
runner reached Liverpool. Prints GL_RENDERER to stderr.
Exit codes as gles_run.c: 1 bad args, 2 file, 3 EGL, 4 compile/link,
5 map, 6 GL error, 7 workgroup too large.
"""
import ctypes as C
import os
import sys

EGL_PLATFORM_GBM_KHR, EGL_PLATFORM_SURFACELESS_MESA = 0x31D7, 0x31DD
EGL_NONE, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_OPENGL_ES_API = 0x3038, 0x3040, 0x40, 0x30A0
EGL_CONTEXT_MAJOR_VERSION, EGL_CONTEXT_MINOR_VERSION = 0x3098, 0x30FB
GL_COMPUTE_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS = 0x91B9, 0x8B81, 0x8B82
GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_READ = 0x90D2, 0x88E4, 0x88E9
GL_ALL_BARRIER_BITS, GL_MAP_READ_BIT, GL_RENDERER, GL_VERSION = 0xFFFFFFFF, 0x0001, 0x1F01, 0x1F02
GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, GL_COMPUTE_WORK_GROUP_SIZE = 0x90EB, 0x8267


def fn(lib, name, res, args):
    f = getattr(lib, name)
    f.restype, f.argtypes = res, args
    return f


def die(code, msg):
    sys.stderr.write(msg + "\n")
    sys.exit(code)


def parse(argv):
    a = dict(srcs=[], groups=1, ins=[], outs=[], ints=[], compile_only=False)
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--compile-only":
            a["compile_only"] = True
            i += 1
            continue
        if i + 1 >= len(argv):
            die(1, f"bad arg {k}")
        v = argv[i + 1]
        if k == "--src":
            a["srcs"].append(v)
        elif k == "--groups":
            a["groups"] = int(v)
        elif k == "--in":
            b, p = v.split(":", 1)
            a["ins"].append((int(b), p))
        elif k == "--out":
            b, rest = v.split(":", 1)
            p, n = rest.rsplit(":", 1)
            a["outs"].append((int(b), p, int(n)))
        elif k == "--int":
            n, val = v.split("=", 1)
            a["ints"].append((n, int(val)))
        else:
            die(1, f"bad arg {k}")
        i += 2
    return a


def slurp(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        die(2, f"cannot open {path}")


def main():
    a = parse(sys.argv[1:])
    egl = C.CDLL("libEGL.so.1")
    gl = C.CDLL("libGLESv2.so.2")
    vp, u, i32 = C.c_void_p, C.c_uint, C.c_int

    drm = os.environ.get("GLES_DRM_DEVICE")
    fd, gbm, gbm_dev = -1, None, None
    if drm:
        fd = os.open(drm, os.O_RDWR | os.O_CLOEXEC)
        gbm = C.CDLL("libgbm.so.1", use_errno=True)
        gbm_dev = fn(gbm, "gbm_create_device", vp, [i32])(fd)
        if not gbm_dev:
            die(3, f"gbm_create_device({drm}) failed")
        plat, native = EGL_PLATFORM_GBM_KHR, gbm_dev
    else:
        plat, native = EGL_PLATFORM_SURFACELESS_MESA, None

    getproc = fn(egl, "eglGetProcAddress", vp, [C.c_char_p])
    p = getproc(b"eglGetPlatformDisplayEXT")
    if not p:
        die(3, "eglGetPlatformDisplayEXT missing")
    dpy = C.CFUNCTYPE(vp, u, vp, vp)(p)(plat, native, None)
    if not dpy:
        die(3, "eglGetPlatformDisplayEXT failed")
    maj, mnr = i32(), i32()
    if not fn(egl, "eglInitialize", u, [vp, C.POINTER(i32), C.POINTER(i32)])(dpy, C.byref(maj), C.byref(mnr)):
        die(3, "eglInitialize failed")
    fn(egl, "eglBindAPI", u, [u])(EGL_OPENGL_ES_API)
    cfg_attr = (i32 * 3)(EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE)
    cfg, ncfg = vp(), i32()
    fn(egl, "eglChooseConfig", u, [vp, C.POINTER(i32), C.POINTER(vp), i32, C.POINTER(i32)])(
        dpy, cfg_attr, C.byref(cfg), 1, C.byref(ncfg))
    ctx_attr = (i32 * 5)(EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE)
    get_error = fn(egl, "eglGetError", i32, [])
    ctx = fn(egl, "eglCreateContext", vp, [vp, vp, vp, C.POINTER(i32)])(
        dpy, cfg if ncfg.value else None, None, ctx_attr)
    if not ctx:
        die(3, "eglCreateContext failed 0x%x" % get_error())
    if not fn(egl, "eglMakeCurrent", u, [vp, vp, vp, vp])(dpy, None, None, ctx):
        die(3, "eglMakeCurrent failed 0x%x" % get_error())

    gs = fn(gl, "glGetString", C.c_char_p, [u])
    sys.stderr.write("GL_RENDERER: %s | GL_VERSION: %s\n" % (gs(GL_RENDERER).decode(), gs(GL_VERSION).decode()))

    src = b"#version 320 es\n" + b"".join(slurp(s) + b"\n" for s in a["srcs"])
    sh = fn(gl, "glCreateShader", u, [u])(GL_COMPUTE_SHADER)
    srcp = (C.c_char_p * 1)(src)
    fn(gl, "glShaderSource", None, [u, i32, C.POINTER(C.c_char_p), vp])(sh, 1, srcp, None)
    fn(gl, "glCompileShader", None, [u])(sh)
    ok = i32()
    fn(gl, "glGetShaderiv", None, [u, u, C.POINTER(i32)])(sh, GL_COMPILE_STATUS, C.byref(ok))
    if not ok.value:
        log = C.create_string_buffer(8192)
        fn(gl, "glGetShaderInfoLog", None, [u, i32, vp, C.c_char_p])(sh, 8192, None, log)
        die(4, "compile:\n" + log.value.decode(errors="replace"))
    prog = fn(gl, "glCreateProgram", u, [])()
    fn(gl, "glAttachShader", None, [u, u])(prog, sh)
    fn(gl, "glLinkProgram", None, [u])(prog)
    get_prog = fn(gl, "glGetProgramiv", None, [u, u, C.POINTER(i32)])
    get_prog(prog, GL_LINK_STATUS, C.byref(ok))
    if not ok.value:
        log = C.create_string_buffer(8192)
        fn(gl, "glGetProgramInfoLog", None, [u, i32, vp, C.c_char_p])(prog, 8192, None, log)
        die(4, "link:\n" + log.value.decode(errors="replace"))

    if a["compile_only"]:
        inv, wg = i32(), (i32 * 3)()
        fn(gl, "glGetIntegerv", None, [u, C.POINTER(i32)])(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, C.byref(inv))
        get_prog(prog, GL_COMPUTE_WORK_GROUP_SIZE, wg)
        print("compiled OK; local size %dx%dx%d; GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS %d" % (wg[0], wg[1], wg[2], inv.value))
        sys.stdout.flush()
        return 0 if wg[0] * wg[1] * wg[2] <= inv.value else 7

    fn(gl, "glUseProgram", None, [u])(prog)
    loc = fn(gl, "glGetUniformLocation", i32, [u, C.c_char_p])
    uni = fn(gl, "glUniform1i", None, [i32, i32])
    for name, val in a["ints"]:
        uni(loc(prog, name.encode()), val)

    gen = fn(gl, "glGenBuffers", None, [i32, C.POINTER(u)])
    bind = fn(gl, "glBindBuffer", None, [u, u])
    data = fn(gl, "glBufferData", None, [u, C.c_ssize_t, vp, u])
    base = fn(gl, "glBindBufferBase", None, [u, u, u])
    keep, outs = [], []
    for b, path in a["ins"]:
        blob = slurp(path)
        buf = u()
        gen(1, C.byref(buf))
        bind(GL_SHADER_STORAGE_BUFFER, buf)
        cbuf = C.create_string_buffer(blob, len(blob))
        keep.append(cbuf)
        data(GL_SHADER_STORAGE_BUFFER, len(blob), C.cast(cbuf, vp), GL_STATIC_DRAW)
        base(GL_SHADER_STORAGE_BUFFER, b, buf)
    for b, path, n in a["outs"]:
        buf = u()
        gen(1, C.byref(buf))
        bind(GL_SHADER_STORAGE_BUFFER, buf)
        data(GL_SHADER_STORAGE_BUFFER, n, None, GL_DYNAMIC_READ)
        base(GL_SHADER_STORAGE_BUFFER, b, buf)
        outs.append((buf, path, n))

    fn(gl, "glDispatchCompute", None, [u, u, u])(a["groups"], 1, 1)
    fn(gl, "glMemoryBarrier", None, [u])(GL_ALL_BARRIER_BITS)
    fn(gl, "glFinish", None, [])()
    mapr = fn(gl, "glMapBufferRange", vp, [u, C.c_ssize_t, C.c_ssize_t, u])
    unmap = fn(gl, "glUnmapBuffer", u, [u])
    gl_error = fn(gl, "glGetError", u, [])
    for buf, path, n in outs:
        bind(GL_SHADER_STORAGE_BUFFER, buf)
        ptr = mapr(GL_SHADER_STORAGE_BUFFER, 0, n, GL_MAP_READ_BIT)
        if not ptr:
            die(5, "map failed 0x%x" % gl_error())
        with open(path, "wb") as f:
            f.write(C.string_at(ptr, n))
        unmap(GL_SHADER_STORAGE_BUFFER)
    err = gl_error()
    if err:
        die(6, "GL error 0x%x" % err)

    fn(egl, "eglMakeCurrent", u, [vp, vp, vp, vp])(dpy, None, None, None)
    fn(egl, "eglDestroyContext", u, [vp, vp])(dpy, ctx)
    fn(egl, "eglTerminate", u, [vp])(dpy)
    if gbm_dev:
        fn(gbm, "gbm_device_destroy", None, [vp])(gbm_dev)
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
