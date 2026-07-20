from __future__ import annotations

import os

from finetune_last2_alignn import main


if __name__ == "__main__":
    os.environ.setdefault("DGLBACKEND", "pytorch")
    main()
