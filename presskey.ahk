#NoEnv  			; Recommended for performance and compatibility with future AutoHotkey releases.
#SingleInstance, Force

if A_Args.Length() < 1 or A_Args.Length() > 1
{
    MsgBox % "This script requires 1 parameter but it received " A_Args.Length() "."
    ExitApp
}


KeySendDelay := 0		; Sets value(ms) for delay between send key commands.
KeyPressDuration := 10	; Sets value(ms) for duration each key press is held {Down}.
setkeydelay, %KeySendDelay%, %KeyPressDuration% 		; Sets delay(ms) between keystrokes issued. Arguments are delay between keystrokes and press duration, respectively.
if A_Args[1] = "w" {
    Send, {w down}
    Send, {w up}
}
else if A_Args[1] = "s"
    SendRaw, s
else if A_Args[1] = "a"
    SendRaw, {a}
else if A_Args[1] = "d"
    SendRaw, {d}
Return
