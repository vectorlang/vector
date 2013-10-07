type operator = Add | Sub | Mul | Div | Mod
    | Lshift | Rshift
    | Less | LessEq | Greater | GreaterEq | Eq | NotEq
    | BitAnd | BitXor | BitOr | LogAnd | LogOr ;;

type expr =
    Binop of expr * operator * expr
  | Assign of string * expr
  | IntLit of int
  | Ident of string ;;
