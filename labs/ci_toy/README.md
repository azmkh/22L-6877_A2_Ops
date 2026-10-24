# Task 6.6 (toy) - GitHub Actions in a throwaway repository

1. Create a new **empty** GitHub repository, e.g. `ROLLNO_A2_Ops_ci_toy`.
2. Copy the contents of this folder (including the hidden `.github` folder) into it:
   ```powershell
   git clone https://github.com/<you>/ROLLNO_A2_Ops_ci_toy.git $HOME\ci_toy
   Copy-Item -Recurse -Force .\labs\ci_toy\* $HOME\ci_toy\
   Copy-Item -Recurse -Force .\labs\ci_toy\.github $HOME\ci_toy\
   cd $HOME\ci_toy; git add .; git commit -m "Toy CI"; git push
   ```
3. **Red run:** in `calc.py` change `return a + b` to `return a - b`, commit, push. The Actions tab shows
   a failed run for both Python versions in the matrix. Screenshot it.
4. **Green run:** revert the change, commit, push. Screenshot the passing run.
