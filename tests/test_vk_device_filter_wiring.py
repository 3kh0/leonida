"""Every runtime program that chooses a Vulkan GPU must honour LEONIDA_VK_DEVICE.

docs/testing.md ("Choosing the GPU"): on a machine with several GPUs, LEONIDA_VK_DEVICE names the one to test, and a
value that matches no device fails the run instead of testing another GPU. A program that ranks the devices with
``device_rank`` / ``vulkan_device_rank`` alone takes the best-ranked GPU whatever the variable says:
vk_descriptor_bench and vk_sandbox did that (#5781).

The check reads the sources, so it runs without a GPU: a file that enumerates or ranks physical devices must filter
them (``pick_device`` or ``vulkan_device_wanted``) and have a failure path that names the filter
(``report_no_device`` or ``vulkan_device_filter()``).
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = [ROOT / "runtime" / d for d in ("src", "shims", "test")]

# The helpers themselves and their CPU unit tests.
HELPERS = {
    "runtime/src/vulkan_device_filter.h",
    "runtime/test/vk_device_pick.h",
    "runtime/test/vulkan_device_filter_test.cpp",
}

# Fixed by an open pull request: drop the entry when it lands. #2647 (TEST-469) moves vk_bindless_test onto
# pick_device / report_no_device.
PENDING = {
    "runtime/test/vk_bindless_test.cpp",
}

SELECTS = re.compile(r"\bvkEnumeratePhysicalDevices\b|\bdevice_rank\s*\(|\bvulkan_device_rank\s*\(")
FILTERS = re.compile(r"\bpick_device\s*\(|\bvulkan_device_wanted\s*\(")
FAILS = re.compile(r"\breport_no_device\s*\(|\bvulkan_device_filter\s*\(\s*\)")


def device_choosers():
    """Repository-relative paths of the runtime sources that enumerate or rank Vulkan physical devices."""
    found = []
    for path in sorted(p for d in SOURCE_DIRS for p in d.rglob("*")):
        if path.suffix not in {".cpp", ".h", ".hpp", ".cc", ".mm"} or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in HELPERS:
            continue
        if SELECTS.search(path.read_text(encoding="utf-8", errors="replace")):
            found.append(rel)
    return found


def unfiltered(paths):
    """[(path, what is missing)] for the sources that skip the filter or its failure path."""
    wrong = []
    for rel in paths:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        if not FILTERS.search(text):
            wrong.append((rel, "no pick_device / vulkan_device_wanted"))
        elif not FAILS.search(text):
            wrong.append((rel, "no report_no_device / vulkan_device_filter() failure path"))
    return wrong


class DeviceFilterWiringTest(unittest.TestCase):
    def test_the_scan_finds_the_device_choosers(self):
        found = set(device_choosers())
        # A pattern that stops matching would pass every file; these choose a device and must stay in the scan.
        for rel in ("runtime/test/vk_compute_test.cpp", "runtime/test/vk_sampler_test.cpp",
                    "runtime/test/vk_test_device.h", "runtime/shims/d3d12_vulkan_device.cpp",
                    "runtime/test/vk_descriptor_bench.cpp", "runtime/test/vk_sandbox.cpp"):
            self.assertIn(rel, found)

    def test_every_device_chooser_honours_leonida_vk_device(self):
        wrong = unfiltered(p for p in device_choosers() if p not in PENDING)
        self.assertEqual(wrong, [], "rank devices through vk_device_pick.h (pick_device, report_no_device), "
                                    "or skip !vulkan_device_wanted(name) and fail naming vulkan_device_filter()")

    def test_a_rank_only_chooser_is_caught(self):
        # The loop vk_sandbox had before #5781's fix.
        rank_only = ("for (auto p : pds) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);\n"
                     "    if (int r = leonida::vk_test::device_rank(pp.deviceType, pp.vendorID); r > best) pd = p; }\n")
        self.assertTrue(SELECTS.search(rank_only))
        self.assertFalse(FILTERS.search(rank_only))


if __name__ == "__main__":
    unittest.main()
