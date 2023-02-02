use liquid::model::KString;
use liquid::partials::PartialCompiler;
use liquid::{ParserBuilder, ValueView};
use liquid_core::{
    Display_filter, Expression, Filter, FilterParameters, FilterReflection, FromFilterParameters,
    ParseFilter, ParseTag, Renderable, Runtime, TagReflection, Value,
};

pub fn register<C: PartialCompiler>(parser: ParserBuilder<C>) -> ParserBuilder<C> {
    parser.tag(AmxTag).filter(LeftShift).filter(Setting).filter(Unsigned)
}

pub fn globals() -> Vec<(KString, Value)> {
    vec![
        ("AMX_SET".to_string().into(), Value::scalar(amx_nop_op_imm5(17, 0))),
        ("AMX_CLR".to_string().into(), Value::scalar(amx_nop_op_imm5(17, 1))),
    ]
}

fn amx_nop_op_imm5(op: usize, imm5: usize) -> String {
    format!("nop\nnop\nnop\n.word 0x{:x}\n", (0x201000 + (op << 5) + imm5))
}

fn amx_nop_op_gpr(op: usize, gpr: usize) -> String {
    format!(".word 0x{:x}", (0x201000 + (op << 5) + gpr))
}

#[derive(Copy, Clone)]
struct AmxTag;

impl ParseTag for AmxTag {
    fn reflection(&self) -> &dyn liquid_core::TagReflection {
        self
    }

    fn parse(
        &self,
        mut arguments: liquid_core::TagTokenIter,
        _options: &liquid_core::Language,
    ) -> liquid_core::Result<Box<dyn liquid_core::Renderable>> {
        let op = arguments.expect_next("expects op and gpr")?.as_str().to_string();
        let gpr = arguments
            .expect_next("expects op and gpr")?
            .as_str()
            .trim_start_matches('x')
            .parse::<usize>()
            .unwrap();
        let op_id = [
            "ldx", "ldy", "stx", "sty", "ldz", "stz", "ldzi", "stzi", "extrx", "extry", "fma64",
            "fms64", "fma32", "fms32", "mac16", "fma16", "fms16", "setclr", "vecint", "vecfp",
            "matint", "matfp", "genlut",
        ]
        .iter()
        .position(|x| x == &op)
        .unwrap();
        Ok(Box::new(RenderedAmxTag(format!(
            "{} \t\t\t\t// AMX {op} x{gpr}\n",
            amx_nop_op_gpr(op_id, gpr)
        ))))
    }
}

impl TagReflection for AmxTag {
    fn tag(&self) -> &str {
        "amx"
    }

    fn description(&self) -> &str {
        "translate to an Apple AMX instruction"
    }
}

#[derive(Clone, Debug)]
struct RenderedAmxTag(String);

impl Renderable for RenderedAmxTag {
    fn render_to(
        &self,
        writer: &mut dyn std::io::Write,
        _runtime: &dyn liquid_core::Runtime,
    ) -> liquid_core::Result<()> {
        writer.write_all(self.0.as_bytes()).unwrap();
        Ok(())
    }
}

#[derive(Debug, FilterParameters)]
struct ShiftArgs {
    #[parameter(description = "The number to shift the input by.")]
    operand: Expression,
}

#[derive(Clone, ParseFilter, FilterReflection)]
#[filter(
    name = "lsl",
    description = "Shift left a number by the given operand.",
    parameters(ShiftArgs),
    parsed(LeftShiftFilter)
)]
struct LeftShift;

#[derive(Debug, FromFilterParameters, Display_filter)]
#[name = "lsl"]
struct LeftShiftFilter {
    #[parameters]
    args: ShiftArgs,
}

impl Filter for LeftShiftFilter {
    fn evaluate(&self, input: &dyn ValueView, runtime: &dyn Runtime) -> liquid_core::Result<Value> {
        let args = self.args.evaluate(runtime)?;

        let operand = args
            .operand
            .as_scalar()
            .ok_or_else(|| invalid_argument("operand", "Number expected"))?;

        let result = input
            .as_scalar()
            .unwrap()
            .to_integer()
            .and_then(|i| operand.to_integer().map(|o| Value::scalar(i << o)))
            .ok_or_else(|| invalid_argument("operand", "Integer expected"))?;

        Ok(result)
    }
}

#[derive(Clone, ParseFilter, FilterReflection)]
#[filter(
    name = "setting",
    description = "Set the bit deigned by the operand.",
    parameters(ShiftArgs),
    parsed(SettingFilter)
)]
struct Setting;

#[derive(Debug, FromFilterParameters, Display_filter)]
#[name = "setting"]
struct SettingFilter {
    #[parameters]
    args: ShiftArgs,
}

impl Filter for SettingFilter {
    fn evaluate(&self, input: &dyn ValueView, runtime: &dyn Runtime) -> liquid_core::Result<Value> {
        let args = self.args.evaluate(runtime)?;

        let operand = args
            .operand
            .as_scalar()
            .ok_or_else(|| invalid_argument("operand", "Number expected"))?;

        let result = input
            .as_scalar()
            .unwrap()
            .to_integer()
            .and_then(|i| operand.to_integer().map(|o| Value::scalar(i | (1 << o))))
            .ok_or_else(|| invalid_argument("operand", "Integer expected"))?;

        Ok(result)
    }
}

fn invalid_argument<S>(argument: S, cause: S) -> liquid::Error
where
    S: Into<liquid_core::model::KString>,
{
    liquid_core::Error::with_msg("Invalid argument")
        .context("argument", argument)
        .context("cause", cause)
}

#[derive(Clone, ParseFilter, FilterReflection)]
#[filter(name = "u", description = "unsigned number", parsed(UnsignedFilter))]
pub struct Unsigned;

#[derive(Debug, Default, Display_filter)]
#[name = "float16"]
struct UnsignedFilter;

impl Filter for UnsignedFilter {
    fn evaluate(
        &self,
        input: &dyn ValueView,
        _runtime: &dyn Runtime,
    ) -> liquid_core::Result<Value> {
        let input = input.as_scalar().unwrap().to_integer().unwrap() as u64;
        Ok(input.to_string().to_value())
    }
}
