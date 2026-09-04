import shutil
import sys


def main() -> int:
    print(f"python        {sys.version.split()[0]}")
    try:
        import torch
    except Exception as exc:
        print(f"torch         import failed: {exc}")
        return 1
    print(f"torch         {torch.__version__}")
    print(f"cuda runtime  {torch.version.cuda}")
    print(f"cuda usable   {torch.cuda.is_available()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        memory = props.total_memory / 1024 ** 3
        print(f"cuda:{index} {props.name} sm_{major}{minor} {memory:.1f} GiB")
    for binary in ("mmseqs", "hmmscan"):
        path = shutil.which(binary)
        print(f"{binary:<13} {path or 'MISSING'}")
    for module in ("transformers", "esm", "antiberty", "abnumber", "parasail"):
        try:
            __import__(module)
        except Exception:
            print(f"{module:<13} MISSING")
        else:
            print(f"{module:<13} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
