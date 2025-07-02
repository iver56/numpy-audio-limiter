* Bump version in `pyproject.toml`
* Update CHANGELOG.md and insert only the changelog for the latest version in README.md
* Commit and push the change with a commit message like this: "Release vx.y.z" (replace x.y.z with the package version)
* Wait for build workflow in GitHub Actions to complete
* Download wheels-all artifact from the build workflow
* Place all the fresh whl files in dist/
* `python -m twine upload dist/*`
* Add a tag with name "x.y.z" to the commit
* Go to https://github.com/iver56/numpy-audio-limiter/releases and create a release where you choose the new tag
