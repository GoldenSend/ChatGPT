# Python Stub File for the Cave Engine's EVENT API
# The code here is available at cave.event. submodule
# This is intended to be used with cave.getEvents()

# Mouse Keys:
MOUSE_LEFT   : int # Left Mouse Button
MOUSE_MIDDLE : int # Middle Mouse Button
MOUSE_RIGHT  : int # Right Mouse Button
# Note: for Mouse Scroll, check cave.Event's getMouseScroll()!

# Controller (Joystick):
BUTTON_A : int
BUTTON_B : int
BUTTON_X : int
BUTTON_Y : int
BUTTON_BACK : int
BUTTON_GUIDE : int
BUTTON_START : int
BUTTON_LEFTSTICK : int
BUTTON_RIGHTSTICK : int
BUTTON_LEFTSHOULDER : int
BUTTON_RIGHTSHOULDER : int
BUTTON_DPAD_UP : int
BUTTON_DPAD_DOWN : int
BUTTON_DPAD_LEFT : int
BUTTON_DPAD_RIGHT : int
BUTTON_MISC1 : int # Xbox Series X share button, PS5 microphone button, Nintendo Switch Pro capture button, Amazon Luna microphone button
BUTTON_PADDLE1 : int # Xbox Elite paddle P1 (upper left, facing the back)
BUTTON_PADDLE2 : int # Xbox Elite paddle P3 (upper right, facing the back)
BUTTON_PADDLE3 : int # Xbox Elite paddle P2 (lower left, facing the back)
BUTTON_PADDLE4 : int # Xbox Elite paddle P4 (lower right, facing the back)
BUTTON_TOUCHPAD : int # PS4/PS5 touchpad button

# Alphabet Keys:
KEY_A : int # A
KEY_B : int # B
KEY_C : int # C
KEY_D : int # D
KEY_E : int # E
KEY_F : int # F
KEY_G : int # G
KEY_H : int # H
KEY_I : int # I
KEY_J : int # J
KEY_K : int # K
KEY_L : int # L
KEY_M : int # M
KEY_N : int # N
KEY_O : int # O
KEY_P : int # P
KEY_Q : int # Q
KEY_R : int # R
KEY_S : int # S
KEY_T : int # T
KEY_U : int # U
KEY_V : int # V
KEY_W : int # W
KEY_X : int # X
KEY_Y : int # Y
KEY_Z : int # Z

# Numeric Keys:
KEY_0 : int # 0
KEY_1 : int # 1
KEY_2 : int # 2
KEY_3 : int # 3
KEY_4 : int # 4
KEY_5 : int # 5
KEY_6 : int # 6
KEY_7 : int # 7
KEY_8 : int # 8
KEY_9 : int # 9

# Function Keys (F):
KEY_F1 : int # F1
KEY_F2 : int # F2
KEY_F3 : int # F3
KEY_F4 : int # F4
KEY_F5 : int # F5
KEY_F6 : int # F6
KEY_F7 : int # F7
KEY_F8 : int # F8
KEY_F9 : int # F9
KEY_F10 : int # F10
KEY_F11 : int # F11
KEY_F12 : int # F12
KEY_F13 : int # F13
KEY_F14 : int # F14
KEY_F15 : int # F15
KEY_F16 : int # F16
KEY_F17 : int # F17
KEY_F18 : int # F18
KEY_F19 : int # F19
KEY_F20 : int # F20
KEY_F21 : int # F21
KEY_F22 : int # F22
KEY_F23 : int # F23
KEY_F24 : int # F24

# Keyboard Arrow Keys:
KEY_UP : int # Up
KEY_DOWN : int # Down
KEY_LEFT : int # Left
KEY_RIGHT : int # Right

# Shift Keys:
KEY_LSHIFT : int # Left Shift
KEY_RSHIFT : int # Right Shift

# Alt Keys:
KEY_LALT : int # Left Alt
KEY_RALT : int # Right Alt

# Ctrl Keys:
KEY_LCTRL : int # Left Ctrl
KEY_RCTRL : int # Right Ctrl

# Return Keys:
KEY_RETURN  : int # Return
KEY_RETURN2 : int # Return2

# Tab Key:
KEY_TAB : int # Tab

# Volume Keys:
KEY_VOLUMEDOWN : int # Volume Down
KEY_VOLUMEUP   : int # Volume Up

# Other Audio Related Keys:
KEY_AUDIOMUTE : int # Mute Volume
KEY_AUDIOPLAY : int # Play Media
KEY_AUDIONEXT : int # Next Track
KEY_AUDIOPREV : int # Previous Track
KEY_AUDIOSTOP : int # Stop Media

# Keypad Keys:
KEY_KP_000 : int # Keypad 000
KEY_KP_00 : int # Keypad 00
KEY_KP_0 : int # Keypad 0
KEY_KP_1 : int # Keypad 1
KEY_KP_2 : int # Keypad 2
KEY_KP_3 : int # Keypad 3
KEY_KP_4 : int # Keypad 4
KEY_KP_5 : int # Keypad 5
KEY_KP_6 : int # Keypad 6
KEY_KP_7 : int # Keypad 7
KEY_KP_8 : int # Keypad 8
KEY_KP_9 : int # Keypad 9
KEY_KP_A : int # Keypad A
KEY_KP_B : int # Keypad B
KEY_KP_C : int # Keypad C
KEY_KP_D : int # Keypad D
KEY_KP_E : int # Keypad E
KEY_KP_F : int # Keypad F
KEY_KP_AT : int # Keypad At
KEY_KP_AMPERSAND : int # Keypad Ampersand
KEY_KP_BACKSPACE : int # Keypad Backspace
KEY_KP_BINARY : int # Keypad Binary
KEY_KP_CLEAR : int # Keypad Clear
KEY_KP_CLEARENTRY : int # Keypad Clear Entry
KEY_KP_COLON : int # Keypad Colon
KEY_KP_COMMA : int # Keypad Comma
KEY_KP_DBLAMPERSAND : int # Keypad Dblampersand
KEY_KP_DBLVERTICALBAR : int # Keypad Dblverticalbar
KEY_KP_DECIMAL : int # Keypad Decimal
KEY_KP_DIVIDE : int # Keypad Divide
KEY_KP_ENTER : int # Keypad Enter
KEY_KP_EQUALS : int # Keypad Equals
KEY_KP_EQUALSAS400 : int # Keypad Equalsas400
KEY_KP_EXCLAM : int # Keypad Exclam
KEY_KP_GREATER : int # Keypad Greater
KEY_KP_HASH : int # Keypad Hash
KEY_KP_HEXADECIMAL : int # Keypad Hexadecimal
KEY_KP_LEFTBRACE : int # Keypad Leftbrace
KEY_KP_LEFTPAREN : int # Keypad Leftparen
KEY_KP_LESS : int # Keypad Less
KEY_KP_MEMADD : int # Keypad Memadd
KEY_KP_MEMCLEAR : int # Keypad Memclear
KEY_KP_MEMDIVIDE : int # Keypad Memdivide
KEY_KP_MEMMULTIPLY : int # Keypad Memmultiply
KEY_KP_MEMRECALL : int # Keypad Memrecall
KEY_KP_MEMSTORE : int # Keypad Memstore
KEY_KP_MEMSUBTRACT : int # Keypad Memsubtract
KEY_KP_MINUS : int # Keypad Minus
KEY_KP_MULTIPLY : int # Keypad Multiply
KEY_KP_OCTAL : int # Keypad Octal
KEY_KP_PERCENT : int # Keypad Percent
KEY_KP_PERIOD : int # Keypad Period
KEY_KP_PLUS : int # Keypad Plus
KEY_KP_PLUSMINUS : int # Keypad Plusminus
KEY_KP_POWER : int # Keypad Power
KEY_KP_RIGHTBRACE : int # Keypad Rightbrace
KEY_KP_RIGHTPAREN : int # Keypad Rightparen
KEY_KP_SPACE : int # Keypad Space
KEY_KP_TAB : int # Keypad Tab
KEY_KP_VERTICALBAR : int # Keypad Verticalbar
KEY_KP_XOR : int # Keypad Xor

# Brackets and Parenthesis Keys:
KEY_LEFTBRACKET  : int # Left Bracket "{"
KEY_RIGHTBRACKET : int # Right Bracket "}"
KEY_LEFTPAREN  : int # Left Parenthesis "("
KEY_RIGHTPAREN : int # Right Parenthesis ")"

# Application Control Keypad (ac) Keys:
KEY_AC_BACK : int # Ac Back
KEY_AC_BOOKMARKS : int # Ac Bookmarks
KEY_AC_FORWARD : int # Ac Forward
KEY_AC_HOME : int # Ac Home
KEY_AC_REFRESH : int # Ac Refresh
KEY_AC_SEARCH : int # Ac Search
KEY_AC_STOP : int # Ac Stop

# Other Keys:
KEY_LESS : int # Less
KEY_LGUI : int # Lgui
KEY_MAIL : int # Mail
KEY_MEDIASELECT : int # Media Select
KEY_MENU : int # Menu
KEY_MINUS : int # Minus
KEY_MODE : int # Mode
KEY_MUTE : int # Mute
KEY_NUMLOCKCLEAR : int # Num Lock Clear
KEY_OPER : int # Oper
KEY_OUT : int # Out
KEY_PAGEDOWN : int # Page Down
KEY_PAGEUP : int # Page Up
KEY_PASTE : int # Paste
KEY_PAUSE : int # Pause
KEY_PERCENT : int # Percent
KEY_PERIOD : int # Period
KEY_PLUS : int # Plus
KEY_POWER : int # Power
KEY_PRINTSCREEN : int # Print Screen
KEY_PRIOR : int # Prior
KEY_QUESTION : int # Question
KEY_QUOTE : int # Quote
KEY_QUOTEDBL : int # Quotedbl
KEY_RGUI : int # Rgui
KEY_SCROLLLOCK : int # Scroll Lock
KEY_SELECT : int # Select
KEY_SEMICOLON : int # Semicolon
KEY_SEPARATOR : int # Separator
KEY_SIZE : int # Size
KEY_SLASH : int # Slash
KEY_SLEEP : int # Sleep
KEY_SPACE : int # Space
KEY_STOP : int # Stop
KEY_SYSREQ : int # Sysreq
KEY_THOUSANDSSEPARATOR : int # Thousands Separator
KEY_UNDO : int # Undo
KEY_UNKNOWN : int # Unknown
KEY_WWW : int # Www
KEY_AGAIN : int # Again (Redo)
KEY_ALTERASE : int # Alterase
KEY_AMPERSAND : int # Ampersand
KEY_APPLICATION : int # Application / Compose / Context Menu (Windows) Key
KEY_ASTERISK : int # Asterisk
KEY_AT : int # At
KEY_BACKQUOTE : int # Backquote
KEY_BACKSLASH : int # Backslash
KEY_BACKSPACE : int # Backspace
KEY_BRIGHTNESSDOWN : int # Brightnessdown
KEY_BRIGHTNESSUP : int # Brightnessup
KEY_CALCULATOR : int # Calculator
KEY_CANCEL : int # Cancel
KEY_CAPSLOCK : int # Capslock
KEY_CARET : int # Caret
KEY_CLEAR : int # Clear
KEY_CLEARAGAIN : int # Clearagain
KEY_COLON : int # Colon
KEY_COMMA : int # Comma
KEY_COMPUTER : int # Computer
KEY_COPY : int # Copy
KEY_CRSEL : int # Crsel
KEY_CURRENCYSUBUNIT : int # Currencysubunit
KEY_CURRENCYUNIT : int # Currencyunit
KEY_CUT : int # Cut
KEY_DECIMALSEPARATOR : int # Decimalseparator
KEY_DELETE : int # Delete
KEY_DISPLAYSWITCH : int # Displayswitch
KEY_DOLLAR : int # Dollar
KEY_EJECT : int # Eject
KEY_END : int # End
KEY_EQUALS : int # Equals
KEY_ESCAPE : int # Escape
KEY_EXCLAIM : int # Exclaim
KEY_EXECUTE_ : int # Execute
KEY_EXSEL : int # Exsel
KEY_FIND : int # Find
KEY_GREATER : int # Greater
KEY_HASH : int # Hash
KEY_HELP : int # Help
KEY_HOME : int # Home
KEY_INSERT : int # Insert
KEY_KBDILLUMDOWN : int # Kbdillumdown
KEY_KBDILLUMTOGGLE : int # Kbdillumtoggle
KEY_KBDILLUMUP : int # Kbdillumup