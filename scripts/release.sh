#!/usr/bin/env bash
# Release script — standardizes the release process.
#
# Usage:
#   ./scripts/release.sh 0.1.3
#
# What it does:
#   1. Updates CHANGELOG.md (moves [Unreleased] → [0.1.3])
#   2. Commits and pushes
#   3. Creates and pushes a v0.1.3 tag
#   4. The tag triggers .github/workflows/release.yml which:
#      - Generates release notes from commits
#      - Creates the GitHub Release
#      - Builds + pushes Docker images to GHCR + Docker Hub

set -euo pipefail

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.3"
    exit 1
fi

TAG="v${VERSION}"
DATE=$(date +%Y-%m-%d)
REPO=$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')

echo "=== Preparing release ${TAG} ==="

# Check clean working tree
if ! git diff-index --quiet HEAD --; then
    echo "Error: working tree has uncommitted changes. Commit or stash them first."
    exit 1
fi

# Check CHANGELOG exists
if [ ! -f CHANGELOG.md ]; then
    echo "Error: CHANGELOG.md not found."
    exit 1
fi

# Update CHANGELOG — replace [Unreleased] with version
if grep -q "## \[Unreleased\]" CHANGELOG.md; then
    sed -i "s/## \[Unreleased\]/## \[Unreleased\]\n\n## [${VERSION}] — ${DATE}/" CHANGELOG.md
    echo "Updated CHANGELOG.md"

    # Add version link at bottom
    PREV_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    if [ -n "$PREV_TAG" ]; then
        echo "[${VERSION}]: https://github.com/${REPO}/compare/${PREV_TAG}...${TAG}" >> CHANGELOG.md
    else
        echo "[${VERSION}]: https://github.com/${REPO}/releases/tag/${TAG}" >> CHANGELOG.md
    fi
else
    echo "Warning: [Unreleased] section not found in CHANGELOG.md"
fi

# Commit CHANGELOG
git add CHANGELOG.md
git commit -m "chore: bump CHANGELOG for ${TAG}"

# Create and push tag
git tag -a "${TAG}" -m "${TAG}"
git push
git push --tags

echo ""
echo "=== Done ==="
echo "Tag ${TAG} pushed. The release workflow will:"
echo "  1. Create GitHub Release at https://github.com/${REPO}/releases/tag/${TAG}"
echo "  2. Build + push Docker images to GHCR + Docker Hub"
echo ""
echo "Monitor progress: https://github.com/${REPO}/actions"
