# Linux container sandbox policy

The service runs as uid 10001. Generated Python is additionally isolated by
bubblewrap with read-only runtime/corpus mounts and private namespaces. It fails
closed if the host refuses namespaces. No host privileges are granted to the service.

Docker's default seccomp policy denies the namespace syscalls bubblewrap requires.
Use the supplied policy instead of disabling seccomp or adding SYS_ADMIN:

```sh
docker run --rm -p 127.0.0.1:8000:8000 \
  --security-opt seccomp=deploy/seccomp-bubblewrap.json \
  -e ANTHROPIC_API_KEY -e RNSR_SERVICE_TOKEN \
  -e RNSR_SERVICE_CORPUS_ROOT=/data/corpora \
  -v "$PWD/corpora:/data/corpora:ro" \
  -v "$PWD/runs:/data/runs" rnsr:latest
```

A read-only corpus mount disables persistent annotation writes. Mount that
specific corpus directory writable if annotations are required; generated Python
still receives read-only mounts and uses the parent annotation broker.

This profile derives from the Apache-2.0 Moby default profile retrieved on
2026-09-17. The upstream source SHA-256 is `785b2429264afba4d594320337cb17f144f3c7d51585f9805eef72e28f4f9334`.
It retains the upstream default-deny policy and adds `clone`, `unshare`, `setns`,
`mount`, `umount2`, and `pivot_root` for bubblewrap setup. Kernel capabilities
still scope those operations to the caller's namespaces. No `--privileged`,
`SYS_ADMIN`, host PID/network namespaces, or unconfined seccomp is needed.

Sources: [Moby profile](https://github.com/moby/profiles/blob/main/seccomp/default.json),
[Docker seccomp documentation](https://docs.docker.com/engine/security/seccomp/),
[bubblewrap](https://github.com/containers/bubblewrap).

The host must permit unprivileged user namespaces (and, on AppArmor hosts,
allow bubblewrap to create them). Keep security policy changes specific to the
bubblewrap executable/container; do not disable host-wide protection.
