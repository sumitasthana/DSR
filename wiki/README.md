# wiki

Everything published to the [project wiki](https://github.com/sumitasthana/DSR/wiki).

`pages/` is generated. Do not edit it by hand, and do not edit the wiki in the
browser either: both get overwritten on the next build. The sources are the
operational documents in `day-zero/`, which stay out of this repository.

```
python wiki/build.py           # regenerate pages/
python wiki/build.py --push    # regenerate, then push to the wiki
```

The build fails rather than publishing if a page still points at a file that is
not published.
