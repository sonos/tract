use crate::ast::*;
use tract_core::internal::*;
use tract_core::itertools::Itertools;

macro_rules! comma_loop {
    ($self:ident, $rec: ident, $items: expr) => {
        for (ix, l) in $items.iter().enumerate() {
            if ix > 0 {
                write!($self.w, ", ")?;
            }
            $self.$rec(l)?;
        }
    };
}

pub struct Dumper<'a> {
    w: &'a mut dyn std::io::Write,
}

impl<'a> Dumper<'a> {
    pub fn new(w: &'a mut dyn std::io::Write) -> Dumper {
        Dumper { w }
    }
    pub fn document(&mut self, document: &Document) -> TractResult<()> {
        writeln!(self.w, "version {};\n", document.version)?;
        for ext in document.extension.iter().sorted() {
            writeln!(self.w, "extension {};", ext.join(" "))?;
        }
        if document.extension.len() > 0 {
            writeln!(self.w, "")?;
        }
        self.fragments(&document.fragments)?;
        self.graph_def(&document.graph_def)?;
        Ok(())
    }

    pub fn fragments(&mut self, defs: &[FragmentDef]) -> TractResult<()> {
        for fragment_def in defs.iter().sorted_by_key(|frag| &frag.decl.id) {
            self.fragment_def(&fragment_def)?
        }
        Ok(())
    }

    fn fragment_def(&mut self, def: &FragmentDef) -> TractResult<()> {
        self.fragment_decl(&def.decl)?;
        if let Some(body) = &def.body {
            writeln!(self.w, "{{")?;
            for assignment in body {
                self.assignment(assignment)?;
            }
            writeln!(self.w, "}}\n")?;
        } else {
            writeln!(self.w, ";")?;
        };
        Ok(())
    }

    fn fragment_decl(&mut self, decl: &FragmentDecl) -> TractResult<()> {
        write!(self.w, "fragment {}", decl.id)?;
        if let Some(generic_decl) = &decl.generic_decl {
            if let Some(name) = generic_decl {
                write!(self.w, "<?=")?;
                self.type_name(name)?;
                write!(self.w, ">")?;
            } else {
                write!(self.w, "<?>")?;
            }
        }
        self.parameter_list(&decl.parameters)?;
        write!(self.w, " -> (")?;
        for (ix, res) in decl.results.iter().enumerate() {
            if ix > 0 {
                write!(self.w, ", ")?;
            }
            write!(self.w, "{}: ", res.id)?;
            self.type_spec(&res.spec)?;
        }
        writeln!(self.w, ")")?;
        Ok(())
    }

    fn parameter_list(&mut self, parameters: &[Parameter]) -> TractResult<()> {
        write!(self.w, "(")?;
        for (ix, param) in parameters.iter().enumerate() {
            if ix > 0 {
                write!(self.w, ",")?;
            }
            write!(self.w, "\n    ")?;
            write!(self.w, "{}: ", param.id)?;
            self.type_spec(&param.spec)?;
            if let Some(lit) = &param.lit {
                write!(self.w, " = ")?;
                self.literal(lit)?;
            }
        }
        write!(self.w, "\n)")?;
        Ok(())
    }

    fn type_name(&mut self, name: &TypeName) -> TractResult<()> {
        let s = match name {
            &TypeName::Integer => "integer",
            &TypeName::Scalar => "scalar",
            &TypeName::Logical => "logical",
            &TypeName::String => "string",
            &TypeName::Any => "?",
        };
        write!(self.w, "{}", s)?;
        Ok(())
    }

    fn type_spec(&mut self, spec: &TypeSpec) -> TractResult<()> {
        match spec {
            TypeSpec::Array(t) => {
                self.type_spec(&t)?;
                write!(self.w, "[]")?;
            }
            TypeSpec::Single(s) => self.type_name(s)?,
            TypeSpec::Tensor(t) => {
                write!(self.w, "tensor<")?;
                self.type_name(t)?;
                write!(self.w, ">")?;
            }
            TypeSpec::Tuple(types) => {
                write!(self.w, "(")?;
                comma_loop!(self, type_spec, types);
                write!(self.w, ")")?;
            }
        }
        Ok(())
    }

    fn literal(&mut self, lit: &Literal) -> TractResult<()> {
        match lit {
            Literal::Array(lits) => {
                write!(self.w, "[")?;
                comma_loop!(self, literal, lits);
                write!(self.w, "]")?;
            }
            Literal::Logical(b) => write!(self.w, "{}", if *b { "true" } else { "false" })?,
            Literal::Numeric(num) => write!(self.w, "{}", num)?,
            Literal::String(s) => write!(self.w, "{:?}", s)?,
            Literal::Tuple(lits) => {
                write!(self.w, "(")?;
                comma_loop!(self, literal, lits);
                write!(self.w, ")")?;
            }
        }
        Ok(())
    }

    fn graph_def(&mut self, def: &GraphDef) -> TractResult<()> {
        writeln!(
            self.w,
            "graph {}( {} ) -> ( {} ) {{",
            def.id,
            def.parameters.join(", "),
            def.results.join(", ")
        )?;
        for assignment in &def.body {
            self.assignment(assignment)?;
        }
        writeln!(self.w, "}}")?;
        Ok(())
    }

    fn assignment(&mut self, assignment: &Assignment) -> TractResult<()> {
        write!(self.w, "  ")?;
        self.lvalue(&assignment.left)?;
        write!(self.w, " = ")?;
        self.rvalue(&assignment.right)?;
        writeln!(self.w, ";")?;
        Ok(())
    }

    fn lvalue(&mut self, left: &LValue) -> TractResult<()> {
        match left {
            LValue::Identifier(s) => write!(self.w, "{}", s)?,
            LValue::Tuple(s) => {
                write!(self.w, "( ")?;
                comma_loop!(self, lvalue, s);
                write!(self.w, " )")?;
            }
            LValue::Array(s) => {
                write!(self.w, "[ ")?;
                comma_loop!(self, lvalue, s);
                write!(self.w, " ]")?;
            }
        }
        Ok(())
    }

    fn rvalue(&mut self, rv: &RValue) -> TractResult<()> {
        match rv {
            RValue::Array(vals) => {
                write!(self.w, "[")?;
                comma_loop!(self, rvalue, vals);
                write!(self.w, "]")?;
            }
            RValue::Binary(left, op, right) => {
                write!(self.w, "(")?;
                self.rvalue(&left)?;
                write!(self.w, " {} ", op)?;
                self.rvalue(&right)?;
                write!(self.w, ")")?;
            }
            RValue::Comprehension(comp) => self.comprehension(comp)?,
            RValue::Identifier(id) => write!(self.w, "{}", id)?,
            RValue::IfThenElse(ifte) => {
                self.rvalue(&ifte.then)?;
                write!(self.w, " if ")?;
                self.rvalue(&ifte.cond)?;
                write!(self.w, " else ")?;
                self.rvalue(&ifte.otherwise)?;
            }
            RValue::Invocation(inv) => self.invocation(&inv)?,
            RValue::Literal(lit) => self.literal(lit)?,
            RValue::Subscript(left, s) => {
                self.rvalue(left)?;
                write!(self.w, "[")?;
                match s.as_ref() {
                    Subscript::Single(s) => self.rvalue(s)?,
                    Subscript::Range(a, b) => {
                        if let Some(it) = a {
                            self.rvalue(it)?;
                        }
                        write!(self.w, ":")?;
                        if let Some(it) = b {
                            self.rvalue(it)?;
                        }
                    }
                }
                write!(self.w, "]")?;
            }
            RValue::Tuple(vals) => {
                write!(self.w, "(")?;
                comma_loop!(self, rvalue, vals);
                write!(self.w, ")")?;
            }
            RValue::Unary(op, rv) => {
                write!(self.w, "{}", op)?;
                self.rvalue(rv)?;
            }
        }
        Ok(())
    }

    fn invocation(&mut self, inv: &Invocation) -> TractResult<()> {
        write!(self.w, "{}", inv.id)?;
        if let Some(tn) = &inv.generic_type_name {
            write!(self.w, "<")?;
            self.type_name(tn)?;
            write!(self.w, ">")?;
        }
        write!(self.w, "(")?;
        for (ix, arg) in inv.arguments.iter().enumerate() {
            if ix > 0 {
                write!(self.w, ", ")?;
            }
            if let Some(n) = &arg.id {
                write!(self.w, "{} = ", n)?;
            }
            self.rvalue(&arg.rvalue)?;
        }
        write!(self.w, ")")?;
        Ok(())
    }

    fn comprehension(&mut self, comp: &Comprehension) -> TractResult<()> {
        write!(self.w, "[ for")?;
        for iter in &comp.loop_iters {
            write!(self.w, "{} in ", &iter.0)?;
            self.rvalue(&iter.1)?;
        }
        if let Some(filter) = &comp.filter {
            write!(self.w, " if ")?;
            self.rvalue(filter)?;
        }
        write!(self.w, " yield ")?;
        self.rvalue(&comp.yields)?;
        write!(self.w, "]")?;
        Ok(())
    }
}
