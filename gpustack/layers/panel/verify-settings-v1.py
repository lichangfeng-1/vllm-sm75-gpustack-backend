# -*- coding: utf-8 -*-
"""静态校验 v8 面板 overlay：external 模式只放回设置（外观/助手/模型连接），
standalone 模式保留全部页面与六个设置标签。

用法：python3 verify-settings-v1.py app.js [index.html]
校验通过打印 PANEL_SETTINGS_VERIFY_OK；失败打印 FAIL 并退出 1。
"""
import io
import sys


def need(text, needle, count=1):
    got = text.count(needle)
    if got != count:
        raise SystemExit("FAIL: expected %d occurrence(s), got %d: %s" % (count, got, needle))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: verify-settings-v1.py app.js [index.html]")
    app = io.open(sys.argv[1], encoding="utf-8").read()

    # external 分支：入口早退，且只隐藏模型库/运行配置
    need(app, "if (!externalEngineMode) return;")
    need(app, 'if (["models", "profiles"].includes(el.dataset.page)) el.hidden = true;')
    need(app, 'if (["models", "profiles", "settings"].includes(el.dataset.page)) el.hidden = true;', 0)
    need(app, 'if (externalEngineMode && ["models", "profiles"].includes(id)) id = "tests";')
    need(app, 'if (externalEngineMode && ["models", "profiles", "settings"].includes(id)) id = "tests";', 0)

    # external 只保留三个工作台安全标签，并默认模型连接
    need(app, 'const externalSettingsTabs = new Set(["appearance", "assistant", "connections"]);')
    need(app, 'tab.hidden = !externalSettingsTabs.has(tab.dataset.settingsTab);')
    need(app, 'if (!externalSettingsTabs.has(selectedTab)) selectSettingsTab("connections");')

    # standalone 仍保留六个标签；external 才裁剪
    need(app, "const allowed = externalEngineMode")
    need(app, '? new Set(["appearance", "assistant", "connections"])')
    need(app, 'new Set(["appearance", "assistant", "connections", "sampling", "deployment", "security"])')
    need(app, 'settingsTab = allowed.has(id) ? id : (externalEngineMode ? "connections" : "deployment");')
    need(app, 'settingsTab = allowed.has(id) ? id : "deployment";', 0)

    if len(sys.argv) > 2:
        html = io.open(sys.argv[2], encoding="utf-8").read()
        need(html, 'id="openUnifiedSettings"')
        need(html, "data-settings-tab=", 6)
        for tab in ("appearance", "assistant", "connections", "sampling", "deployment", "security"):
            need(html, 'data-settings-tab="%s"' % tab)
    print("PANEL_SETTINGS_VERIFY_OK")


if __name__ == "__main__":
    main()
