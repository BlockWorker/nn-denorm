`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 26.02.2024 11:37:26
// Design Name: 
// Module Name: float_multiplier
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module float_multiplier #(
        parameter EXP_WIDTH = 8,
        parameter SFD_WIDTH = 7
    ) (
        input [EXP_WIDTH+SFD_WIDTH:0] a,
        input [EXP_WIDTH+SFD_WIDTH:0] b,
        output [EXP_WIDTH+SFD_WIDTH:0] out
    );
    
    localparam EXP_BIAS = {1'b0, {EXP_WIDTH-1{1'b1}}};
    localparam MIN_RELEVANT_EXPSUM = EXP_BIAS - SFD_WIDTH - 1; //any exponent sum below this guarantees underflow to zero
    localparam MAX_RELEVANT_EXPSUM = {1'b0, {EXP_WIDTH-1{1'b1}}, 1'b0} + EXP_BIAS; //any exponent sum above this guarantees overflow to infinity
    
    wire a_sign = a[EXP_WIDTH+SFD_WIDTH];
    wire [EXP_WIDTH-1:0] a_exp = a[EXP_WIDTH+SFD_WIDTH-1:SFD_WIDTH];
    wire [SFD_WIDTH-1:0] a_sfd = a[SFD_WIDTH-1:0];
    
    wire b_sign = b[EXP_WIDTH+SFD_WIDTH];
    wire [EXP_WIDTH-1:0] b_exp = b[EXP_WIDTH+SFD_WIDTH-1:SFD_WIDTH];
    wire [SFD_WIDTH-1:0] b_sfd = b[SFD_WIDTH-1:0];
    
    reg out_sign;
    reg [EXP_WIDTH-1:0] out_exp;
    reg [SFD_WIDTH-1:0] out_sfd;
    
    assign out = {out_sign, out_exp, out_sfd};
    
    wire a_exc = &a_exp; //inputs exceptional (inf or nan)
    wire b_exc = &b_exp;
    wire a_sub = ~(|a_exp); //inputs subnormal
    wire b_sub = ~(|b_exp);
    wire a_zero = ~(|(a[EXP_WIDTH+SFD_WIDTH-1:0])); //inputs zero
    wire b_zero = ~(|(b[EXP_WIDTH+SFD_WIDTH-1:0]));
    wire a_one = (a_exp == EXP_BIAS) & ~(|a_sfd); //inputs one (positive or negative)
    wire b_one = (b_exp == EXP_BIAS) & ~(|b_sfd);
    
    wire [EXP_WIDTH-1:0] a_exp_true = a_exp | a_sub; //true exponents (corrected for subnormal)
    wire [EXP_WIDTH-1:0] b_exp_true = b_exp | b_sub;
    
    wire [SFD_WIDTH:0] a_fsfd = {~a_sub, a_sfd}; //full significands with explicit leading bits
    wire [SFD_WIDTH:0] b_fsfd = {~b_sub, b_sfd};
    
    wire [2*SFD_WIDTH+1:0] fsfd_prod = a_fsfd * b_fsfd; //raw product of full significands
    wire [2*SFD_WIDTH+1:0] prod_stickies;
    assign prod_stickies[0] = 1'b0; //sticky in case of all shift out
    generate genvar i; //generate stickies for each shift value
        for (i = 1; i <= 2*SFD_WIDTH+1; i++) assign prod_stickies[i] = prod_stickies[i-1] | fsfd_prod[i-1];
    endgenerate
    reg [$clog2(2*SFD_WIDTH+3)-1:0] prod_zero_count;
    reg [$clog2(2*SFD_WIDTH+3)-1:0] shift_amt;
    reg [2*SFD_WIDTH+1:0] shifted_intermediate;
    reg [SFD_WIDTH+2:0] shifted_prod_sfd; //includes explicit leading bit and round+sticky bits
    reg [EXP_WIDTH-1:0] new_exp;
    
    wire [EXP_WIDTH:0] exp_sum = a_exp_true + b_exp_true + fsfd_prod[2*SFD_WIDTH+1]; //sum of exponents, plus one if exponent is all the way left
    
    //nearest-rounded significand product: previously calculated product, +1 if (lowest bit & round bit) or (round bit & sticky bit)
    wire [SFD_WIDTH+1:0] rounded_prod_sfd = {1'b0, shifted_prod_sfd[SFD_WIDTH+2:2]} + (&(shifted_prod_sfd[2:1]) | &(shifted_prod_sfd[1:0]));
    
    always @* begin
        shifted_intermediate = 'b0;
        
        //product normalization handling
        if (~(|(exp_sum[EXP_WIDTH:EXP_WIDTH-1]))) begin //exponent underflow: shift right until we hit min exponent
            new_exp = 'b0; //denormalized or zero in any case
            if (exp_sum < MIN_RELEVANT_EXPSUM) begin //underflow to zero guaranteed: return zero
                shifted_prod_sfd = 'b0;
            end else begin //shift product right and denormalize
                shift_amt = {2'b01, {EXP_WIDTH-1{1'b0}}} - exp_sum - ~fsfd_prod[2*SFD_WIDTH+1] - 2; //required right shifts, one less if we're not on very left of product
                shifted_prod_sfd = (fsfd_prod[2*SFD_WIDTH+1:SFD_WIDTH-1] >> shift_amt) | prod_stickies[shift_amt+SFD_WIDTH]; //shift product right, guard becomes round, round|sticky becomes sticky
            end
        end else if (exp_sum > MAX_RELEVANT_EXPSUM) begin //overflow to infinity guaranteed: return infinity
            shifted_prod_sfd = 'b0;
            new_exp = {EXP_WIDTH{1'b1}};
        end else if (fsfd_prod[2*SFD_WIDTH+1]) begin //we're at the left of the product: no shifts needed
            shifted_prod_sfd = {fsfd_prod[2*SFD_WIDTH+1:SFD_WIDTH], prod_stickies[SFD_WIDTH]}; //guard becomes round, round|sticky becomes sticky
            new_exp = exp_sum - EXP_BIAS;
        end else begin //we're not at the left of the product: shift left until we are
            for (prod_zero_count = 1; prod_zero_count < 2*SFD_WIDTH+2; prod_zero_count++) begin //count zeros (except for last, else we'd be in zero case)
                if (fsfd_prod[2*SFD_WIDTH+1-prod_zero_count]) break; //skip uppermost bit, must be zero else we'd be in previous case
            end
            if (prod_zero_count + EXP_BIAS > exp_sum) begin //normalizing would underflow exponent: shift as far as possible and denormalize
                shift_amt = exp_sum - {2'b01, {EXP_WIDTH-1{1'b0}}} + 1;
                new_exp = 'b0;
            end else begin //exponent is fine: shift until normalized
                shift_amt = prod_zero_count;
                new_exp = exp_sum - EXP_BIAS - prod_zero_count + 1;
            end
            shifted_intermediate = fsfd_prod << shift_amt; //raw shift result
            shifted_prod_sfd = {shifted_intermediate[2*SFD_WIDTH+1:SFD_WIDTH], prod_stickies[shift_amt > SFD_WIDTH ? 0 : SFD_WIDTH - shift_amt]}; //filter down to round bit, get sticky
        end
    
        //output
        out_sign = a_sign ^ b_sign; //sign is always xor of input signs
        if (a_exc || b_exc) begin //infinities/nans
            out_exp = {EXP_WIDTH{1'b1}};
            if ((a_exc && |a_sfd) || (b_exc && |b_sfd)) begin //at least one is nan: return nan
                out_sfd = {{(SFD_WIDTH-1){1'b0}}, 1'b1};
            end else begin //at least one is infinity
                if (a_zero || b_zero) begin //other is zero: return nan
                    out_sfd = {{(SFD_WIDTH-1){1'b0}}, 1'b1};
                end else begin //anything else: return infinity
                    out_sfd = 'b0;
                end
            end
        end else begin //only actual numbers
            if (a_zero || b_zero) begin //one input zero: return zero
                out_exp = 'b0;
                out_sfd = 'b0;
            end else if (a_one) begin //a is one: return b
                out_exp = b_exp;
                out_sfd = b_sfd;
            end else if (b_one) begin //b is one: return a
                out_exp = a_exp;
                out_sfd = a_sfd;
            end else if (&new_exp) begin //overflow during multiplication itself: infinity
                out_exp = {EXP_WIDTH{1'b1}};
                out_sfd = 'b0;
            end else if (rounded_prod_sfd[SFD_WIDTH+1]) begin //carry-out during rounding: shift right, increase exponent
                if (&(new_exp[EXP_WIDTH-1:1])) begin //increasing exponent would overflow: return infinity
                    out_exp = {EXP_WIDTH{1'b1}};
                    out_sfd = 'b0;
                end else begin //increasing exponent is fine: return shifted/increased results
                    out_exp = new_exp + 1;
                    out_sfd = rounded_prod_sfd[SFD_WIDTH:1];
                end
            end else if (~(|new_exp) && rounded_prod_sfd[SFD_WIDTH]) begin //expecting denormalized, but rounding renormalizes
                out_exp = 'b1;
                out_sfd = rounded_prod_sfd[SFD_WIDTH-1:0];
            end else begin //general case: no overflows or shifts needed, return result
                out_exp = new_exp;
                out_sfd = rounded_prod_sfd[SFD_WIDTH-1:0];
            end
        end
    end
    
endmodule
