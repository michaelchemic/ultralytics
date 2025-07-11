// MainForm.ui.qml
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    width: 1920
    height: 1080

    Rectangle {
        anchors.fill: parent
        color: "#0a1e2f"
    }

    // 顶部栏
    Rectangle {
        id: topBar
        height: parent.height * 0.06
        width: parent.width
        color: "#123456"
        anchors.top: parent.top

        Text {
            anchors.centerIn: parent
            text: "AI管道病害自动检测平台"
            color: "#00f0ff"
            font.pixelSize: 28
        }
    }

    // 左侧导航栏
    Rectangle {
        id: leftNav
        width: parent.width * 0.10 // 10% 宽度
        height: parent.height
        anchors.left: parent.left
        anchors.top: parent.top
        color: "#0f2a3f"

        Column {
            spacing: 20
            anchors.fill: parent
            anchors.margins: 20

            Repeater {
                model: ["数据标定", "视频处理", "AI检测", "批量处理", "生成PDF报告"]
                delegate: Button {
                    text: modelData
                    font.pixelSize: 18
                    background: Rectangle {
                        color: "transparent"
                    }
                    contentItem: Text {
                        text: modelData
                        color: "white"
                        anchors.centerIn: parent
                    }
                }
            }
        }
    }

    // 主显示区域（QWidget 挂载目标）
    Rectangle {
        id: imageContainer
        objectName: "imageContainer"
        anchors.top: topBar.bottom
        anchors.left: leftNav.right
        anchors.bottom: parent.bottom
        anchors.right: rightPanel.left
        color: "#1a2e3d"
        radius: 10
        border.color: "#00caff"
    }

    // 右侧状态区域
    Rectangle {
        id: rightPanel
        width: parent.width * 0.09 // 9% 宽度
        anchors.top: topBar.bottom
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        color: "#1c3e52"
        border.color: "#00caff"
        radius: 8

        Column {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 20

            Text {
                text: "识别状态"
                color: "white"
                font.pixelSize: 18
            }

            ProgressBar {
                value: 0.4
                from: 0
                to: 1
                width: parent.width * 0.9
            }

            Text {
                text: "缺陷类型: 异物入侵\n位置: 2.2m\n严重度: 中"
                color: "white"
            }
        }
    }
}
