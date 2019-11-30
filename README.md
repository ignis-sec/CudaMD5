# CudaMD5
Finding strings that get md5'ed to php type juggleable hash



## Quick info on PHP type juggling

![](https://camo.githubusercontent.com/a65e02829c02723342095c67303f44f92a50d748/68747470733a2f2f692e696d6775722e636f6d2f5752504372446b2e706e67)

[from here](https://github.com/kaaetech/hackermandoc/wiki/php-type-juggling).

## Usage
Some login mechanisms use non-strict checks and are vulnerable to type juggling. This issue is hard to spot without source code because check is being made on the hash itself, and hashes are often salted.
This code aims to MD5 hashes matching the pattern `^0e[0-9]{30}$` in which php non-strict check will attempt to convert string as integer.

If hashes are salted and you know what the salt might me (for example it often is username), change the salt and saltlen in md5.h.


### Dont use debug mode if you arent building on this tool. Debug mode defeats the whole point of using CUDA.



## Known Issues
- If there are two solutions in BLOCKSIZE*THREADSIZE distance away, only the last one will be detected. (Odds are incredibly low, on average they should be ~10M away)
- Instead of `^0e[0-9]{30}$`, `^0*e[0-9]*$` should be used for pattern checking. This will slightly increase the yield.
- Some of the artifacts remain from starting only in debug mode, which can be removed to slightly optimize runtime.
