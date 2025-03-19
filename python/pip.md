# Clear `pip` cache

`pip cache purge`

# Update package

`pip install --upgrade package_name`

# Force reinstall of a package

`pip install <pkgname> --force-reinstall --upgrade --no-cache-dir`

* `--force-reinstall`: This argument forces pip to reinstall the package even if it's already installed. This is useful if you want to ensure that you're getting a fresh installation of the package, possibly to fix a corrupted installation or to replace it with a different version.

* `--upgrade` (or `-U`): This option tells pip to upgrade the package to the latest version available in the repository. If the package is already installed, pip will install an updated version if one is available. If the package is not installed, it works just like a regular install.

* `--no-cache-dir`: By default, pip caches downloaded packages to speed up future installations. The --no-cache-dir option disables this caching mechanism. This means pip will download the package afresh from the repository. This can be helpful if you suspect the cache is corrupted or you want to ensure you're getting the very latest version directly from the repository.