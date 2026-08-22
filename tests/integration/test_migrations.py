"""Migration 執行器。"""

import pytest

from database.migrate import applied, available, pending, upgrade

pytestmark = pytest.mark.integration


async def test_every_migration_file_is_applied():
    versions = {version for version, _ in available()}
    assert versions, "找不到任何 migration 檔"
    assert versions <= await applied()


async def test_upgrade_is_idempotent():
    assert await upgrade() == []
    assert await pending() == []


async def test_versions_sort_in_execution_order():
    versions = [version for version, _ in available()]
    assert versions == sorted(versions), "檔名排序必須等於執行順序"
